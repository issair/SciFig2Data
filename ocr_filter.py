import os
import json
import shutil
import re
import argparse
import cv2
import pytesseract
from PIL import Image


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 3
    )
    return binary


def ocr_image(img, lang="eng+chi_sim"):
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_img, lang=lang)
    return text.strip()


def process_image(img_file, image_dir):
    try:
        img_path = os.path.join(image_dir, img_file)

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        binary = preprocess(img)

        text_normal = ocr_image(binary)

        rotated = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
        text_rotated = ocr_image(rotated)

        combined_text = "\n".join([t for t in [text_normal, text_rotated] if t])

        if combined_text:
            print(f"Recognition successful ({img_file})")
        else:
            print(f"Warning: {img_file} no text recognized")

        return img_file, combined_text if combined_text else "No text recognized"

    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")
        return img_file, f"OCR processing error: {str(e)}"


def process_directory(directory, subdir_name, output_filename):
    image_dir = os.path.join(directory, subdir_name)

    if not os.path.exists(image_dir):
        print(f"Directory does not exist: {image_dir}")
        return None

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print(f"No PNG images in directory: {image_dir}")
        return None

    results = {}

    print(f"Processing directory {directory}/{subdir_name} using Tesseract OCR")
    print(f"Total {len(image_files)} images to process")

    for i, img_file in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {img_file}")
        _, text = process_image(img_file, image_dir)
        results[img_file] = text

    output_file = os.path.join(directory, output_filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Directory {directory}/{subdir_name} processing complete, results saved to {output_file}")
    return results


def filter_images(data, condition1_keywords, condition2_keywords):
    matched_images = []

    for image_name, text in data.items():
        cleaned_text = text.lower()
        cleaned_text = re.sub(r'[\n\r\t]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        cond1_met = any(keyword in cleaned_text for keyword in condition1_keywords)
        cond2_met = any(keyword in cleaned_text for keyword in condition2_keywords)

        if cond1_met and cond2_met:
            matched_images.append(image_name)

    return matched_images


def filter_and_copy_figures(directory, figure_data, result2_dir,
                            condition1_keywords, condition2_keywords):
    if not figure_data:
        return

    matched_images = filter_images(figure_data, condition1_keywords, condition2_keywords)

    if matched_images:
        dir_name = os.path.basename(directory)
        target_dir = os.path.join(result2_dir, dir_name)

        extracted_figures_dir = os.path.join(directory, "extracted_figures")
        target_figures_dir = os.path.join(target_dir, "extracted_figures")
        os.makedirs(target_figures_dir, exist_ok=True)

        for image_name in matched_images:
            src_path = os.path.join(extracted_figures_dir, image_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, target_figures_dir)
            else:
                print(f"Warning: Image file does not exist, cannot copy: {src_path}")

        print(f"Filtering complete: successfully copied {len(matched_images)} matching images to {target_figures_dir}")
    else:
        print(f"No images matching the filter criteria in directory {directory}.")


def parse_keywords(value):
    return [kw.strip() for kw in value.split(',') if kw.strip()]


def main():
    parser = argparse.ArgumentParser(description="OCR figure/table images and filter by two keyword groups.")
    parser.add_argument('--condition1-keywords', '-c1', type=parse_keywords, default=["dose", "concentration"], help='Comma-separated keywords for condition 1')
    parser.add_argument('--condition2-keywords', '-c2', type=parse_keywords, default=["viability", "death", "atp"], help='Comma-separated keywords for condition 2')
    parser.add_argument('--base-dir', default='result', help='Base directory to scan (default: result)')
    parser.add_argument('--result2-dir', default='result_filtered', help='Output directory for filtered figures (default: result_filtered)')
    args = parser.parse_args()

    condition1_keywords = args.condition1_keywords
    condition2_keywords = args.condition2_keywords
    base_dir = args.base_dir
    result2_dir = args.result2_dir

    print(f"Condition 1 keywords: {condition1_keywords}")
    print(f"Condition 2 keywords: {condition2_keywords}")

    os.makedirs(result2_dir, exist_ok=True)

    directories = []
    for root, dirs, files in os.walk(base_dir):
        if 'extracted_figures' in dirs or 'extracted_tables' in dirs:
            directories.append(root)

    if not directories:
        print(f"No directories containing extracted_figures or extracted_tables subdirectories found in {base_dir}")
    else:
        print(f"Found {len(directories)} directories to process:")
        for i, d in enumerate(directories):
            print(f"{i+1}. {d}")

        for directory in directories:
            print(f"\n==================== Starting to process directory: {directory} ====================")

            figure_data = process_directory(directory, 'extracted_figures', "figure_text_E.json")

            process_directory(directory, 'extracted_tables', "table_text_E.json")

            if figure_data:
                print(f"Filtering and copying figures for {directory}...")
                filter_and_copy_figures(
                    directory, figure_data, result2_dir,
                    condition1_keywords, condition2_keywords
                )


if __name__ == '__main__':
    main()