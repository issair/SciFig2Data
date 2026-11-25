import os
import cv2
import torch
import argparse
import fitz  # PyMuPDF
import numpy as np
from doclayout_yolo import YOLOv10
from PIL import Image
import io

def optimize_image_size(img_bgr):
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    output = io.BytesIO()
    pil_img.save(
        output,
        format='PNG',
        dpi=(300, 300),
        optimize=True,
        compress_level=3
    )

    return output.getvalue()


def process_pdf(args):
    os.makedirs(args.figures_dir, exist_ok=True)

    tables_dir = args.tables_dir
    os.makedirs(tables_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLOv10(args.model)
    pdf_document = fitz.open(args.pdf_path)

    figure_counter = 1
    table_counter = 1

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        zoom_detect = args.dpi / 72
        pix_detect = page.get_pixmap(matrix=fitz.Matrix(zoom_detect, zoom_detect), alpha=False)
        img_np_detect = np.frombuffer(pix_detect.samples, dtype=np.uint8).reshape(pix_detect.height, pix_detect.width, 3)
        img_bgr_detect = cv2.cvtColor(img_np_detect, cv2.COLOR_RGB2BGR)

        zoom_crop = args.dpi / 108
        pix_crop = page.get_pixmap(matrix=fitz.Matrix(zoom_crop, zoom_crop), alpha=False)
        img_np_crop = np.frombuffer(pix_crop.samples, dtype=np.uint8).reshape(pix_crop.height, pix_crop.width, 3)
        img_bgr_crop = cv2.cvtColor(img_np_crop, cv2.COLOR_RGB2BGR)

        scale_x = pix_crop.width / pix_detect.width
        scale_y = pix_crop.height / pix_detect.height

        det_res = model.predict(
            img_bgr_detect,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
        )

        if not det_res or len(det_res) == 0:
            print(f"No detections on page {page_num+1}")
            continue

        names = model.names

        figure_idx = next((idx for idx, name in names.items() if "figure" in name.lower()), None)
        table_idx = next((idx for idx, name in names.items() if "table" in name.lower() and "caption" not in name.lower()), None)

        if figure_idx is None:
            print("Warning: 'figure' class not found")
        if table_idx is None:
            print("Warning: 'table' class not found")

        orig_h, orig_w = img_bgr_crop.shape[:2]

        for det in det_res:
            for box in det.boxes:
                cls_idx = int(box.cls)
                conf = float(box.conf)

                is_figure = (figure_idx is not None and cls_idx == figure_idx)
                is_table = (table_idx is not None and cls_idx == table_idx)

                if not (is_figure or is_table):
                    continue

                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(float)

                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                expand_pixels = 10
                x1 = max(0, int(x1 - expand_pixels))
                y1 = max(0, int(y1 - expand_pixels))
                x2 = min(orig_w, int(x2 + expand_pixels))
                y2 = min(orig_h, int(y2 + expand_pixels))

                if x1 >= x2 or y1 >= y2:
                    print(f"Invalid coordinates: ({x1}, {y1}, {x2}, {y2})")
                    continue

                crop_img = img_bgr_crop[y1:y2, x1:x2]

                h, w = crop_img.shape[:2]
                if max(h, w) < 400:
                    print(f"Skipped small region ({w}x{h}) on page {page_num+1}")
                    continue

                if page_num == 0:
                    print(f"Skipping page 1 region ({x1}, {y1}, {x2}, {y2})")
                    continue

                optimized_img = optimize_image_size(crop_img)

                if is_figure:
                    filename = f"figure_{figure_counter}_page_{page_num+1}_conf_{conf:.2f}.png"
                    save_path = os.path.join(args.figures_dir, filename)
                    with open(save_path, "wb") as f:
                        f.write(optimized_img)
                    print(f"Saved figure: {save_path}")
                    figure_counter += 1

                if is_table:
                    filename = f"table_{table_counter}_page_{page_num+1}_conf_{conf:.2f}.png"
                    save_path = os.path.join(tables_dir, filename)
                    with open(save_path, "wb") as f:
                        f.write(optimized_img)
                    print(f"Saved table: {save_path}")
                    table_counter += 1

    print(f"\nProcessing completed:")
    print(f"- Figures saved in: {args.figures_dir}")
    print(f"- Tables saved in: {args.tables_dir}")
    print(f"Total figures: {figure_counter - 1}")
    print(f"Total tables: {table_counter - 1}")

def process_single_pdf(args, pdf_path):
    args.pdf_path = pdf_path

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    result_root = f"result/{pdf_name}"

    if os.path.exists(result_root):
        print(f"[SKIP] {pdf_path} already processed")
        return

    args.figures_dir = f"{result_root}/extracted_figures"
    args.tables_dir = f"{result_root}/extracted_tables"

    process_pdf(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF pages with YOLOv10 model')
    parser.add_argument('--model', default='weights/doclayout_yolo_docstructbench_imgsz1024.pt', type=str)
    parser.add_argument('--pdf-path', default=None, type=str)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--figures-dir', type=str)
    parser.add_argument('--tables-dir', type=str)
    parser.add_argument('--imgsz', default=1024, type=int)
    parser.add_argument('--conf', default=0.8, type=float)
    parser.add_argument('--dpi', default=300, type=int)

    args = parser.parse_args()

    if args.pdf_path:
        process_single_pdf(args, args.pdf_path)

    elif args.data_dir:
        pdf_files = sorted([
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.lower().endswith(".pdf")
        ])

        if not pdf_files:
            print(f"No PDF files found in {args.data_dir}")
            exit(0)

        print(f"Found {len(pdf_files)} PDFs in {args.data_dir}")

        for pdf_path in pdf_files:
            print(f"\n=== Processing: {pdf_path} ===")
            process_single_pdf(args, pdf_path)

    else:
        print("Please specify either --pdf-path or --data-dir")
