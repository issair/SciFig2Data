import os
import json
import re
import argparse
import pytesseract
import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms as T
import cv2
import numpy as np
from collections import defaultdict


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]


def get_model(model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=False)
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(device)


def run_classification(directory, model, config, device, args):
    input_dir = os.path.join(directory, "separated_subfigures")
    output_index_json = os.path.join(directory, "classification_chart.json")

    os.makedirs(os.path.dirname(output_index_json), exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Skipping classification: separated_subfigures folder not found")
        return None

    num_classes = len(config['label_to_idx'])
    idx_to_label = {v: k for k, v in config['label_to_idx'].items()}

    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    image_files.sort(key=natural_sort_key)

    classification_index = []
    stats = {
        "unclassified": 0,
        "total_images": len(image_files)
    }

    print(f"Running model classification, {len(image_files)} images to process...")
    for filename in image_files:
        file_path = os.path.join(input_dir, filename)
        image_class = None

        try:
            image = Image.open(file_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                probs = probabilities.squeeze()

            top_prob, top_index = torch.max(probs, dim=0)
            top_prob = top_prob.item()
            top_index = top_index.item()

            if top_index in idx_to_label:
                label = idx_to_label[top_index]
                if label in ["horizontal_bar", "line", "vertical_bar", "scatter", "scatter-line"] and top_prob >= args.classify_threshold:
                    image_class = label
                    stats[image_class] = stats.get(image_class, 0) + 1
                else:
                    stats["unclassified"] += 1
            else:
                stats["unclassified"] += 1

            if image_class:
                classification_index.append({
                    "filename": filename,
                    "class": image_class
                })

        except Exception as e:
            print(f"Error classifying file {filename}: {str(e)}")
            stats["unclassified"] += 1

    if classification_index:
        sorted_index = sorted(classification_index, key=lambda x: natural_sort_key(x["filename"]))
        with open(output_index_json, "w", encoding='utf-8') as f:
            json.dump(sorted_index, f, ensure_ascii=False, indent=4)

    print(f"Classification complete! Unclassified: {stats['unclassified']}/{stats['total_images']}")
    return stats


def preprocess_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def ocr_single_image(img, config=""):
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_img, config=config)
    return text.strip()


def process_ocr_image(img_file, image_dir):
    try:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        binary = preprocess_ocr(img)
        custom_config = r'--oem 3 -l eng+ell+equ'

        # First pass: recognize original image
        text_normal = ocr_single_image(binary, custom_config)
        # Second pass: recognize rotated 90°
        rotated = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
        text_rotated = ocr_single_image(rotated, custom_config)

        combined_text = "\n".join([t for t in [text_normal, text_rotated] if t])
        return img_file, combined_text if combined_text else "No text recognized"

    except Exception as e:
        print(f"Error during OCR processing of {img_file}: {str(e)}")
        return img_file, f"OCR processing error: {str(e)}"


def should_filter(text, keep_keywords1, keep_keywords2, filter_keywords=None):
    if filter_keywords is None:
        filter_keywords = []

    cleaned_text = text.lower().replace('\n', ' ').replace(' ', '')

    if any(keyword in cleaned_text for keyword in filter_keywords):
        return True

    has_keep1 = any(keyword in cleaned_text for keyword in keep_keywords1)
    has_keep2 = any(keyword in cleaned_text for keyword in keep_keywords2)

    if has_keep1 or has_keep2:
        return False

    return True


def run_ocr_and_filter(directory, keep_keywords1, keep_keywords2, filter_keywords=None):
    image_dir = os.path.join(directory, 'separated_subfigures')
    classification_file = os.path.join(directory, 'classification_chart.json')

    if not os.path.exists(classification_file):
        print(f"Classification file does not exist, skipping OCR: {classification_file}")
        return

    with open(classification_file, 'r', encoding='utf-8') as f:
        chart_data = json.load(f)

    image_files = [item['filename'] for item in chart_data]
    if not image_files:
        print(f"No images in classification file need OCR")
        return

    ocr_results = {}
    print(f"Starting OCR recognition on {len(image_files)} classified images...")

    for i, img_file in enumerate(image_files):
        _, text = process_ocr_image(img_file, image_dir)
        ocr_results[img_file] = text
        print(f"OCR progress {i+1}/{len(image_files)}: {img_file}")

    filtered_data = []
    for item in chart_data:
        img_file = item['filename']
        if img_file in ocr_results:
            if should_filter(ocr_results[img_file], keep_keywords1, keep_keywords2, filter_keywords):
                print(f"Filtering chart: {img_file}")
                continue
        filtered_data.append(item)

    with open(classification_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Directory {directory} processing complete, {len(filtered_data)}/{len(chart_data)} charts retained after filtering")


def parse_keywords(value):
    if not value:
        return []
    return [kw.strip() for kw in value.split(',') if kw.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Integrated script for chart classification and OCR filtering")
    parser.add_argument('--base_dir', type=str, default='result_filtered', help="Base directory")
    parser.add_argument('--checkpoint', type=str, default='weights/swin_base_patch4_window7_224_best.pth', help="Model weights path")
    parser.add_argument('--model_name', type=str, default='swin_base_patch4_window7_224', help="Model architecture name")
    parser.add_argument('--config_path', default='weights/config/swinconfig.json', help="Classification configuration file path")
    parser.add_argument('--classify_threshold', type=float, default=0.7, help="Classification confidence threshold")
    parser.add_argument('--keep-keywords1', '-k1', type=parse_keywords, default=["dose", "concentration"], help='Comma-separated keywords group 1 that should be KEPT')
    parser.add_argument('--keep-keywords2', '-k2', type=parse_keywords, default=["viability", "death", "atp"], help='Comma-separated keywords group 2 that should be KEPT')
    parser.add_argument('--filter-keywords', '-f', type=parse_keywords, default=[], help='Comma-separated keywords that cause immediate FILTERING')

    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    num_classes = len(config['label_to_idx'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_name, num_classes, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loading complete!\n")

    directories = []
    for root, dirs, files in os.walk(args.base_dir):
        if 'separated_subfigures' in dirs:
            directories.append(root)

    print(f"Found {len(directories)} directories to process:")
    for i, d in enumerate(directories):
        print(f"{i+1}. {d}")

    for directory in directories:
        print(f"\n{'=' * 60}")
        print(f"Starting to process directory: {directory}")
        try:
            stats = run_classification(directory, model, config, device, args)

            if stats and stats['total_images'] > 0:
                run_ocr_and_filter(
                    directory,
                    args.keep_keywords1,
                    args.keep_keywords2,
                    args.filter_keywords
                )

        except Exception as e:
            print(f"Error processing directory {directory}: {str(e)}")
            continue