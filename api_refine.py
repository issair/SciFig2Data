import base64
from openai import OpenAI
import os
import re
import glob
import argparse
import fitz
import json
import torch
import cv2
import numpy as np
from doclayout_yolo import YOLOv10
import pytesseract


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def read_markdown_file(md_path):
    with open(md_path, "r", encoding="utf-8") as md_file:
        return md_file.read()


def extract_image_names(md_content):
    lines = md_content.split('\n')
    images = []
    for line in lines:
        if "Extraction Result:" in line:
            img_name = line.split('Extraction Result:')[-1].strip()
            images.append(img_name)
    return images


def correct_image_name(md_image_name):
    pattern = r'(_conf_[0-9.]+)_[0-9.]+_[0-9.]+\.png$'
    corrected_name = re.sub(pattern, r'\1.png', md_image_name)

    if corrected_name == md_image_name:
        pattern2 = r'(_conf_[0-9.]+)_[0-9.]+\.png$'
        corrected_name = re.sub(pattern2, r'\1.png', md_image_name)

        if corrected_name == md_image_name:
            print(f"Warning: Cannot correct image name '{md_image_name}', using original name")

    return corrected_name


def extract_page_numbers_from_figures(figures_dir):
    page_numbers = set()
    pattern = re.compile(r'page_(\d+)')

    for filename in os.listdir(figures_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            match = pattern.search(filename)
            if match:
                page_num = int(match.group(1))
                page_numbers.add(page_num)

    return sorted(page_numbers)


def extract_figure_captions(args):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLOv10(args.model)
    pdf_document = fitz.open(args.pdf_path)
    results = {}

    for page_num in args.pages:
        if page_num < 1 or page_num > len(pdf_document):
            print(f"Warning: Page number {page_num} out of range (1-{len(pdf_document)})")
            continue

        captions = []
        current_page = page_num
        found_captions = False

        while not found_captions and current_page <= len(pdf_document):
            page = pdf_document.load_page(current_page - 1)

            zoom = args.dpi / 108
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            det_res = model.predict(
                img_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
            )

            caption_idx = None
            for idx, name in model.names.items():
                if name == 'figure_caption':
                    caption_idx = idx
                    break

            if caption_idx is None:
                print(f"Page {current_page}: 'figure_caption' class not detected")
                current_page += 1
                continue

            caption_boxes = []
            for det in det_res:
                for box in det.boxes:
                    cls_idx = int(box.cls)
                    conf = float(box.conf)

                    if cls_idx == caption_idx and conf >= args.conf:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        caption_boxes.append({
                            'box': [x1, y1, x2, y2],
                            'conf': conf
                        })

            if not caption_boxes:
                print(f"Page {current_page}: No caption regions detected")
                current_page += 1
                continue

            caption_boxes.sort(key=lambda x: x['box'][1])

            for i, cap_box in enumerate(caption_boxes):
                x1, y1, x2, y2 = cap_box['box']

                x1 = max(0, min(img_bgr.shape[1]-1, x1))
                y1 = max(0, min(img_bgr.shape[0]-1, y1))
                x2 = max(1, min(img_bgr.shape[1], x2))
                y2 = max(1, min(img_bgr.shape[0], y2))

                caption_img = img_bgr[y1:y2, x1:x2]

                gray = cv2.cvtColor(caption_img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                try:
                    text = pytesseract.image_to_string(thresh, lang='chi_sim+eng')
                    captions.append(text.strip())
                    print(f"Page {current_page} - Caption {i+1} OCR successful")
                except Exception as e:
                    print(f"Page {current_page} - Caption {i+1} OCR failed: {str(e)}")
                    captions.append("OCR failed")

            found_captions = True

        if not found_captions:
            print(f"Warning: No captions found on page {page_num} or its next page")
            results[f"page_{page_num}"] = []
        else:
            results[f"page_{page_num}"] = captions

    return results


def extract_page_from_filename(filename):
    pattern = re.compile(r'page_(\d+)')
    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    return 0


def save_markdown_file(content, output_path):
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(content)
    print(f"Markdown file saved to: {output_path}")


def process_directory(target_dir, pdf_path, model_path, api_model, dpi=300, imgsz=[1024], conf=0.6):
    prompt = """
    Task Requirements:
    1. Analyze the provided chart image and identify the chart legends and labels.
    2. Identify configurations and parameters in the caption.
    3. Improve the table structure:
    - Accurately name the columns based on the chart content.
    - Add new columns if necessary, such as time, assay, nanoparticle type, cell type, etc.
    - Add the correct units (extracted from the axis labels).
    4. Output requirements:
    - Preserve the original document structure.
    - Modify only the table sections.
    - Output the optimized Markdown file.
    """

    client = OpenAI(
        api_key="",
        base_url=""
    )

    IMAGE_DIR = os.path.join(target_dir, "extracted_figures/")
    MD_DIR = target_dir

    md_files = glob.glob(os.path.join(MD_DIR, "*.md"))
    if not md_files:
        print(f"No Markdown files found in directory {MD_DIR}")
        return

    md_file_path = md_files[0]
    md_content = read_markdown_file(md_file_path)

    image_names = extract_image_names(md_content)
    if not image_names:
        print("Warning: No image information found in Markdown file")
        return

    class Args:
        pass

    args_caption = Args()
    args_caption.pdf_path = pdf_path
    args_caption.model = model_path
    args_caption.dpi = dpi
    args_caption.imgsz = imgsz
    args_caption.conf = conf
    args_caption.pages = extract_page_numbers_from_figures(IMAGE_DIR)

    print(f"Starting caption extraction, pages involved: {args_caption.pages}")
    captions_map = extract_figure_captions(args_caption)
    print("Caption extraction complete\n")

    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"The content of markdown file:\n{md_content}"}
    ]

    processed_images = set()
    for md_image_name in image_names:
        actual_image_name = correct_image_name(md_image_name)
        if actual_image_name in processed_images:
            continue

        image_path = os.path.join(IMAGE_DIR, actual_image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image file does not exist: {image_path} (original name: {md_image_name})")
            continue

        page_num = extract_page_from_filename(actual_image_name)
        caption_text = ""

        if f"page_{page_num}" in captions_map:
            page_captions = captions_map[f"page_{page_num}"]
            caption_text = "\n".join([f"Caption {i+1}: {cap}" for i, cap in enumerate(page_captions)])
        else:
            print(f"Warning: No caption information for page {page_num}")

        messages_content.append({
            "type": "text",
            "text": f"The caption:\n{caption_text}"
        })

        print(f"Processing image: {md_image_name} → {actual_image_name}, page: {page_num}")
        image_base64 = encode_image_to_base64(image_path)
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

        processed_images.add(actual_image_name)

    print("\nSending request to API...")
    response = client.chat.completions.create(
        model=api_model,
        messages=[{"role": "user", "content": messages_content}],
        max_tokens=50000
    )

    result_content = response.choices[0].message.content
    output_filename = "optimized_report.md"
    output_path = os.path.join(MD_DIR, output_filename)
    save_markdown_file(result_content, output_path)


def main():
    parser = argparse.ArgumentParser(description="Process Markdown files and charts in specified directories")
    parser.add_argument("directories", nargs='+',
                        help="List of directory paths to process (result_filtered/pdf_name/)")
    parser.add_argument("--swin-model", type=str, default='/home/{your_path}/weights/doclayout_yolo_docstructbench_imgsz1024.pt',
                        help="YOLO model path")
    parser.add_argument("--api-model", type=str, default='gpt-4o',
                        help="OpenAI API model name")
    parser.add_argument("--dpi", type=int, default=300, help="Image rendering resolution")
    parser.add_argument("--imgsz", type=int, nargs='+', default=[1024], help="Model input size")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")

    args = parser.parse_args()

    for directory in args.directories:
        target_dir = directory.rstrip('/') + '/'
        pdf_name = os.path.basename(os.path.normpath(directory))
        pdf_path = os.path.join('/home/xhwude/recurrence_g/data', f"{pdf_name}.pdf")
        print(f"\n{'='*50}")
        print(f"Starting to process directory: {target_dir}")
        print(f"Auto-generated PDF path: {pdf_path}")

        try:
            process_directory(
                target_dir=target_dir,
                pdf_path=pdf_path,
                model_path=args.swin_model,
                api_model=args.api_model,
                dpi=args.dpi,
                imgsz=args.imgsz,
                conf=args.conf
            )
            print(f"Directory processing complete: {target_dir}")
        except Exception as e:
            print(f"Error processing directory {target_dir}: {str(e)}")
        finally:
            print(f"{'='*50}\n")


if __name__ == "__main__":
    main()