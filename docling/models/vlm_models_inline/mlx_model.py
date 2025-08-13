import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL.Image import Image

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction, VlmPredictionToken
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BaseVlmPageModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Global lock for MLX model calls - MLX models are not thread-safe
# All MLX models share this lock to prevent concurrent MLX operations
_MLX_GLOBAL_LOCK = threading.Lock()


class HuggingFaceMlxModel(BaseVlmPageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled

        self.vlm_options = vlm_options
        self.max_tokens = vlm_options.max_new_tokens
        self.temperature = vlm_options.temperature

        if self.enabled:
            try:
                from mlx_vlm import generate, load, stream_generate  # type: ignore
                from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
                from mlx_vlm.utils import load_config  # type: ignore
            except ImportError:
                raise ImportError(
                    "mlx-vlm is not installed. Please install it via `pip install mlx-vlm` to use MLX VLM models."
                )

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            self.apply_chat_template = apply_chat_template
            self.stream_generate = stream_generate

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(
                    self.vlm_options.repo_id,
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            ## Load the model
            self.vlm_model, self.processor = load(artifacts_path)
            self.config = load_config(artifacts_path)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, f"vlm-mlx-{self.vlm_options.repo_id}"):
                    assert page.size is not None

                    hi_res_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )
                    if hi_res_image is not None:
                        im_width, im_height = hi_res_image.size

                    # populate page_tags with predicted doc tags
                    page_tags = ""

                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    if callable(self.vlm_options.prompt):
                        user_prompt = self.vlm_options.prompt(page.parsed_page)
                    else:
                        user_prompt = self.vlm_options.prompt
                    prompt = self.apply_chat_template(
                        self.processor, self.config, user_prompt, num_images=1
                    )

                    # MLX models are not thread-safe - use global lock to serialize access
                    with _MLX_GLOBAL_LOCK:
                        _log.debug(
                            "MLX model: Acquired global lock for __call__ method"
                        )
                        start_time = time.time()
                        _log.debug("start generating ...")

                        # Call model to generate:
                        tokens: list[VlmPredictionToken] = []

                        output = ""
                        for token in self.stream_generate(
                            self.vlm_model,
                            self.processor,
                            prompt,
                            [hi_res_image],
                            max_tokens=self.max_tokens,
                            verbose=False,
                            temp=self.temperature,
                        ):
                            if len(token.logprobs.shape) == 1:
                                tokens.append(
                                    VlmPredictionToken(
                                        text=token.text,
                                        token=token.token,
                                        logprob=token.logprobs[token.token],
                                    )
                                )
                            elif (
                                len(token.logprobs.shape) == 2
                                and token.logprobs.shape[0] == 1
                            ):
                                tokens.append(
                                    VlmPredictionToken(
                                        text=token.text,
                                        token=token.token,
                                        logprob=token.logprobs[0, token.token],
                                    )
                                )
                            else:
                                _log.warning(
                                    f"incompatible shape for logprobs: {token.logprobs.shape}"
                                )

                            output += token.text
                            if "</doctag>" in token.text:
                                break

                        generation_time = time.time() - start_time
                        _log.debug("MLX model: Released global lock")
                    page_tags = output

                    _log.debug(
                        f"{generation_time:.2f} seconds for {len(tokens)} tokens ({len(tokens) / generation_time} tokens/sec)."
                    )
                    page.predictions.vlm_response = VlmPrediction(
                        text=page_tags,
                        generation_time=generation_time,
                        generated_tokens=tokens,
                    )

                yield page

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Optional[str] = None,
    ) -> Iterable[VlmPrediction]:
        from mlx_vlm import generate

        # MLX models are not thread-safe - use global lock to serialize access
        with _MLX_GLOBAL_LOCK:
            _log.debug("MLX model: Acquired global lock for thread safety")
            for image in image_batch:
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    if image.ndim == 3 and image.shape[2] in [3, 4]:
                        # RGB or RGBA array
                        from PIL import Image as PILImage

                        image = PILImage.fromarray(image.astype(np.uint8))
                    elif image.ndim == 2:
                        # Grayscale array
                        from PIL import Image as PILImage

                        image = PILImage.fromarray(image.astype(np.uint8), mode="L")
                    else:
                        raise ValueError(
                            f"Unsupported numpy array shape: {image.shape}"
                        )

                # Ensure image is in RGB mode (handles RGBA, L, etc.)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Handle prompt with priority: parameter > vlm_options.prompt > error
                if prompt is not None:
                    user_prompt = prompt
                elif not callable(self.vlm_options.prompt):
                    user_prompt = self.vlm_options.prompt
                else:
                    raise ValueError(
                        "vlm_options.prompt is callable but no prompt parameter provided to process_images. "
                        "Please provide a prompt parameter when calling process_images directly."
                    )

                # Use the MLX chat template approach like in the __call__ method
                formatted_prompt = self.apply_chat_template(
                    self.processor, self.config, user_prompt, num_images=1
                )

                # Generate text from the image - MLX can accept PIL Images directly despite type annotations
                start_time = time.time()
                generated_result = generate(
                    self.vlm_model,
                    self.processor,
                    formatted_prompt,
                    image=image,  # Pass PIL Image directly - much more efficient than disk I/O
                    verbose=False,
                    temp=self.temperature,
                    max_tokens=self.max_tokens,
                )
                generation_time = time.time() - start_time

                # MLX generate returns a tuple (text, info_dict), extract just the text
                if isinstance(generated_result, tuple):
                    generated_text = generated_result[0]
                    _log.debug(
                        f"MLX generate returned tuple with additional info: {generated_result[1] if len(generated_result) > 1 else 'N/A'}"
                    )
                else:
                    generated_text = generated_result

                _log.debug(f"Generated text in {generation_time:.2f}s.")
                yield VlmPrediction(
                    text=generated_text,
                    generation_time=generation_time,
                    # MLX generate doesn't expose tokens directly, so we leave it empty
                    generated_tokens=[],
                )
            _log.debug("MLX model: Released global lock")
