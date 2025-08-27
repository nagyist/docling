import argparse
import json
from pathlib import Path
import shutil

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings

DOC_FILE = "2203.01017v2.pdf"


def main():
    r""" """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
        help="Branch name"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        required=False,
        default="cpu",
        help="Device to run the conversion"
    )
    parser.add_argument(
        "-w", "--work-dir",
        type=Path,
        required=False,
        default="/Users/nli/docling/heron_debugging",
        help="Work directory"
    )
    args = parser.parse_args()

    pdf_path = args.work_dir / DOC_FILE
    print(f"Name: {args.name}")
    print(f"Input file: {pdf_path}")

    # Enable debugging
    settings.debug.visualize_cells = True
    settings.debug.visualize_ocr = True
    settings.debug.visualize_layout = True
    settings.debug.visualize_raw_layout = True
    settings.debug.visualize_tables = True

    # Locally decide the device
    if args.device.lower() == "cpu":
        device = AcceleratorDevice.CPU
    elif args.device.lower() == "mps":
        device = AcceleratorDevice.MPS
    else:
        raise ValueError(f"Unsupported device: {device}")

    # Setup conversion pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.generate_page_images = True
    pipeline_options.accelerator_options.device = device

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PdfFormatOption().backend,
            )
        }
    )

    # Convert
    doc_result = converter.convert(pdf_path)
    doc = doc_result.document

    # Export and save as json
    out_dir = args.work_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fn = out_dir / "2305.03393v1-pg9.json"
    print(f"Out dir: {out_dir}")
    with open(save_fn, "w") as fd:
       dd = doc.export_to_dict()
       json.dump(dd, fd)

    # Move the debug dir
    debug_dir = Path("debug/")
    if debug_dir.is_dir():
        dest_debug = out_dir / "debug"
        if dest_debug.is_dir():
            shutil.rmtree(dest_debug)
        shutil.move(debug_dir, out_dir)
        print(f"")

    # Visualize the document
    viz_imgs = doc.get_visualization()
    for page_no, img in viz_imgs.items():
        if page_no is not None:
            img.save(out_dir / f"docling_p{page_no}.png")


if __name__ == "__main__":
    main()
