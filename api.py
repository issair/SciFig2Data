import base64
from openai import OpenAI
import os
import re
from datetime import datetime
import json
import argparse
import time

client = OpenAI(
    api_key="",
    base_url=""
)

line_prompt = """
# Line Chart Data Precise Extraction Instructions

## 1. Basic Identification
- **Axis Analysis**:
  - Horizontal/vertical axis labels and units
  - Scale range and intervals (for unit conversion reference only)

## 2. Data Point Extraction Rules
- **Key Point Location**:
  - Prioritize using **actual nodes** on the line as data points
  - When no nodes exist, use **uniform sampling** (evenly spaced points based on scale intervals)
- **Baseline Determination**: Use the x-value (horizontal axis value) of data points as the baseline

## 3. Specific Scenario Handling Rules
- **Multi-line Processing**: Distinguish data sequences of different lines

## 4. Output Format Specification
| Series Name | X Value / Unit | Y Value / Unit |

**Format Requirements**:
- X Value: Write numerical values if horizontal axis is numerical, write category names if categorical
- Units only marked in column headers within parentheses, not repeated in individual cells
"""

bar_prompt = """
## 1. Basic Identification
- **Chart Type Identification**: Clearly distinguish between regular bar charts or stacked bar charts
- **Axis Analysis**:
  - Horizontal/vertical axis labels and units
  - Scale range and intervals (for unit conversion reference only)

## 2. Data Point Extraction Rules
- **Key Point Location**: Use the **center of each bar's top** as the data point
- **Baseline Determination**: Use the x-value (horizontal axis value) corresponding to each bar as the baseline

## 3. Specific Scenario Handling Rules
- **Stacked Bar Processing**: Record both individual segment values (stacked portions) and total bar height
- **Error Bar Processing**: Extract error values
- **Horizontal Bar Charts**: Use y-axis values as baseline for data point location and X value extraction

## 4. Output Format Specification
| Series Name | X Value / Unit | Y Value / Unit |

**Format Requirements**:
- X Value: Write numerical values if horizontal axis is numerical, write category names if categorical
- Units only marked in column headers within parentheses, not repeated in individual cells
"""

scatter_prompt = """
# Scatter Plot Data Precise Extraction Instructions

## 1. Basic Identification
- **Axis Analysis**:
  - Horizontal/vertical axis labels and units
  - Scale range and intervals (for unit conversion reference only)

## 2. Data Point Extraction Rules
- **Key Point Location**:
  - Extract all visible scatter points as data points
  - For dense regions, prioritize characteristic distribution points (such as edge points, cluster center points)
- **Baseline Determination**: Use the x-value (horizontal axis value) of data points as the baseline

## 3. Specific Scenario Handling Rules
- **Multi-series Processing**: Distinguish data sequences by different colors/shapes
- **Overlapping Point Processing**: Identify and annotate overlapping or near-overlapping data points

## 4. Output Format Specification
| Series Name | X Value / Unit | Y Value / Unit |

**Format Requirements**:
- X Value: Write numerical values if horizontal axis is numerical, write category names if categorical
- Units only marked in column headers within parentheses, not repeated in individual cells
"""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_error_index(error_list, json_path):
    """Save the error index as a JSON file"""
    if not error_list:
        return None

    error_path = os.path.join(os.path.dirname(json_path), "error.json")
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(error_list, f, indent=4, ensure_ascii=False)

    print(f"📝 Error index file saved: {error_path}")
    return error_path

def process_images_from_classification(json_path, image_dir, work_dir, model_name, recover_mode=False):
    mode_label = "🔧 Recovery Mode" if recover_mode else "🚀 Normal Mode"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            classification_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {json_path}: {str(e)}")
        return None

    if len(classification_data) == 0:
        print(f"⚠ Classification data is empty, skipping processing: {json_path}")
        return None

    report_dir = os.path.dirname(json_path)
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "Recovery" if recover_mode else ""

    work_dir_basename = os.path.basename(work_dir)
    report_filename = os.path.join(report_dir, f"chart_data_extraction_{prefix}report_{work_dir_basename}_{report_timestamp}.md")

    report_content = f"""# 📊 Chart Data Extraction {prefix}Report

## Basic Information
- **Processing Mode**: {mode_label}
- **Generated At**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Images Processed**: {len(classification_data)}
- **Source JSON File**: {os.path.basename(json_path)}
- **Image Directory**: {image_dir}

---

"""

    success_count = 0
    error_files = []
    error_list = []

    for idx, item in enumerate(classification_data):
        filename = item["filename"]
        chart_class = item["class"]
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            error_msg = f"⚠ Image does not exist: {image_path}"
            print(error_msg)
            report_content += f"## ❌ Error: {filename}\n{error_msg}\n\n"
            error_files.append(filename)
            error_list.append({
                "filename": filename,
                "class": chart_class,
                "error": "file not found",
                "timestamp": datetime.now().isoformat()
            })
            continue

        print(f"\n{'=' * 60}")
        print(f"🔍 Processing image [{idx + 1}/{len(classification_data)}]: {filename} | Class: {chart_class}")
        print(f"{'=' * 60}")

        if chart_class == "line" or chart_class == "scatter_line":
            prompt = line_prompt
        elif chart_class == "vertical_bar" or chart_class == "horizontal_bar":
            prompt = bar_prompt
        elif chart_class == "scatter" or chart_class == "vertical_interval" or chart_class == "horizontal_interval":
            prompt = scatter_prompt
        else:
            prompt = line_prompt

        try:
            image_base64 = encode_image_to_base64(image_path)
            img_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=30000
            )

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content

                if '|' in content:
                    pipe_index = content.find('|')
                    extracted_content = content[pipe_index:].strip()
                else:
                    extracted_content = "No table returned"
                
                success_count += 1

                report_content += f"## 📄 Extraction Result: {filename}\n"
                report_content += f"- **Image Class**: `{chart_class}`\n"
                report_content += f"- **Processing Time**: {img_timestamp}\n\n"
                report_content += f"{extracted_content}\n\n---\n\n"

                preview = extracted_content[:50] + "..." if len(extracted_content) > 50 else extracted_content
                print(f"📄 Content preview:\n{preview}\n")
                time.sleep(1)

            else:
                error_msg = f"❌ No valid response received: {filename}"
                print(error_msg)
                report_content += f"## ❌ Error: {filename}\n{error_msg}\n\n"
                error_files.append(filename)
                error_list.append({
                    "filename": filename,
                    "class": chart_class,
                    "error": "no valid response",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            error_msg = f"❌ Error processing image ({filename}): {str(e)}"
            print(error_msg)
            report_content += f"## ❌ Error: {filename}\n{error_msg}\n\n"
            error_files.append(filename)
            error_list.append({
                "filename": filename,
                "class": chart_class,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    report_content += "\n## 📈 Processing Statistics\n"
    report_content += f"- ✅ Successfully processed: {success_count} images\n"
    report_content += f"- ❌ Failed: {len(error_files)} images\n"

    if error_files:
        report_content += "\n### Failed Image List\n"
        for file in error_files:
            report_content += f"- `{file}`\n"

        error_path = save_error_index(error_list, json_path)
        if error_path:
            report_content += f"\n> **Error Index File**: `{os.path.basename(error_path)}`"

    if not recover_mode and error_list:
        report_content += f"\n> 💡 Tip: Error entries found. You can run recovery mode to process these files: \n> `error.json` has been generated in the same directory"

    os.makedirs(os.path.dirname(report_filename), exist_ok=True)

    with open(report_filename, 'w', encoding='utf-8') as report_file:
        report_file.write(report_content)

    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete! Report saved to: {report_filename}")
    print(f"- Success: {success_count} | Failed: {len(error_files)}")
    print(f"{'=' * 60}")

    return report_filename

def process_directory(work_dir, model_name, recover_mode):
    """Process a single directory"""
    print(f"\n{'=' * 60}")
    print(f"{'🔧 Recovery Mode' if recover_mode else '🚀 Normal Mode'} - Processing directory: {work_dir}")
    print(f"{'=' * 60}")

    if recover_mode:
        json_path = os.path.join(work_dir, "error.json")
        if not os.path.exists(json_path):
            print(f"❌ Error: Recovery mode requires error.json file, but not found: {json_path}")
            return None
    else:
        json_path = os.path.join(work_dir, "classification_chart.json")
        if not os.path.exists(json_path):
            print(f"❌ Error: Classification file not found: {json_path}")
            return None

    image_dir = os.path.join(work_dir, "separated_subfigures")
    if not os.path.exists(image_dir):
        print(f"❌ Error: Image directory not found: {image_dir}")
        return None

    return process_images_from_classification(json_path, image_dir, work_dir, model_name, recover_mode)

def main():
    parser = argparse.ArgumentParser(description='Process image classification tasks')
    parser.add_argument('directories', type=str, nargs='+', help='Working directory paths (can be multiple)')
    parser.add_argument('--api-model', type=str, default='gpt-4o',
                    help='Model name to use for the OpenAI API')
    parser.add_argument('--recover', action='store_true', help='Use recovery mode')
    args = parser.parse_args()

    RECOVER_MODE = args.recover
    DIRECTORIES = args.directories
    MODEL_NAME = args.api_model
    
    for work_dir in DIRECTORIES:
        abs_dir = os.path.abspath(work_dir)
        
        try:
            report_path = process_directory(abs_dir, MODEL_NAME,RECOVER_MODE)
            if report_path:
                print(f"✅ Processing complete: {abs_dir}, report path: {report_path}")
            else:
                print(f"❌ Processing failed: {abs_dir}")
        except Exception as e:
            print(f"❌ Error processing directory {abs_dir}: {str(e)}")

if __name__ == "__main__":
    main()