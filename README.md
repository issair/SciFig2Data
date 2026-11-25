# SciFig2Data

## 📖 Overview

**SciFig2Data** is an end-to-end workflow for **recovering structured experimental data from scientific PDF papers**. It combines document layout analysis, OCR-based semantic filtering, compound-figure decomposition, chart classification, and multimodal large language models (MLLMs) to identify relevant quantitative plots, reconstruct their numerical data, and associate the results with experimental context from figure captions.

---

## 🖥️ Online Demo

If you wish to try a practical workflow without setting up the full local environment, an online demo is **coming soon**. It will let you run the core stages of SciFig2Data — from PDF upload to reconstructed chart data — directly in the browser.

👉 [Try the online demo](<DEMO_URL>) (coming soon)

---

## 📦 Model Weights

The required model weights and configuration files are available on the GitHub **Releases** page:

👉 [Download model weights](../../releases/weights)

> **Note:** These weights are third-party assets subject to their original licenses (see [License](#-license)). Download and place them in the `weights/` directory before running the pipeline.

---

## 🔄 Pipeline

SciFig2Data processes scientific PDFs through the following six stages. The scripts can be run sequentially, while `sep.sh` provides a batch entry point for subfigure decomposition across paper directories.

| # | Stage | Script | Environment | Purpose |
|---|---|---|---|---|
| 1 | Document layout detection | `doclayout.py` | `env_doclayout` | Renders PDF pages at multiple resolutions and detects candidate figure and table regions with [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO). |
| 2 | Pre-decomposition semantic filtering | `ocr_filter.py` | `env_doclayout` | Performs OCR on detected regions and retains figures containing keywords from both predefined keyword groups; extracted table text is saved as JSON. |
| 3 | Compound-figure decomposition | `sep.sh` (`sep.py`) | `env_cnntf1` | Batch-processes filtered figures with [figure-separator](https://github.com/apple2373/figure-separator), then refines panel boundaries and restores top-to-bottom, left-to-right reading order. |
| 4 | Panel filtering and chart classification | `classifier_filter.py` | `env_swin` | Applies panel-level OCR filtering (matching either keyword group) and classifies retained panels with [Swin-chart](https://github.com/moured/Swin-chart). |
| 5 | Numerical data reconstruction | `api.py` | `env_doclayout` | Uses chart-specific instructions and an MLLM to reconstruct numerical measurements from classified chart images and generate Markdown reports. |
| 6 | Caption-based metadata integration | `api_refine.py` | `env_doclayout` | Extracts figure captions from the source PDFs and enriches the reconstructed Markdown tables with experimental context. |

---

## 📂 Repository Structure

The repository is organized as follows:

```text
SciFig2Data/
│
├── doclayout.py
├── ocr_filter.py
├── sep.py
├── sep.sh
├── classifier_filter.py
├── api.py
├── api_refine.py
│
├── data/                          # Input PDF papers
│
├── weights/                       # Model weights and configurations
│   ├── doclayout_yolo_docstructbench_imgsz1024.pt
│   ├── figure-sepration-model-submitted-544.pb
│   ├── swin_base_patch4_window7_224_best.pth
│   └── config/
│       ├── yolov3_default_subfig.cfg
│       └── swinconfig.json
│
├── requirements/
│   ├── env_doclayout.txt
│   ├── env_swin.txt
│   └── env_cnntf1.yml
│
├── result/                        # Initially extracted figures and tables
│   └── paper_name/
│       ├── extracted_figures/
│       ├── extracted_tables/
│       ├── figure_text_E.json
│       └── table_text_E.json
│
├── result_filtered/               # Filtered figures and subsequent results
│   └── paper_name/
│       ├── extracted_figures/
│       ├── separated_subfigures/
│       ├── classification_chart.json
│       ├── chart_data_extraction_report_*.md
│       ├── (error.json)
│       └── optimized_report.md

```

---

## ⚙️ Environment Setup

SciFig2Data requires three independent Conda environments to accommodate the dependencies of different processing components.

### 1. Document Layout Analysis and API Processing

Environment: `env_doclayout`

```bash
conda create -n env_doclayout python=3.10.18
conda activate env_doclayout
pip install -r requirements/env_doclayout.txt
```

Used by:

- `doclayout.py`
- `ocr_filter.py`
- `api.py`
- `api_refine.py`

### 2. Chart Classification

Environment: `env_swin`

```bash
conda create -n env_swin python=3.10.18
conda activate env_swin
pip install -r requirements/env_swin.txt
```

Used by:

- `classifier_filter.py`

### 3. Figure Separation

Environment: `env_cnntf1`

```bash
conda env create -f requirements/env_cnntf1.yml
```

Used by:

- `sep.sh (sep.py)`

### 4. Environment-Script Mapping

| Environment | Scripts |
|---|---|
| `env_doclayout` | `doclayout.py`, `ocr_filter.py`, `api.py`, `api_refine.py` |
| `env_cnntf1` | `sep.sh (sep.py)` |
| `env_swin` | `classifier_filter.py` |

---

## 🚀 Usage

The following instructions describe the complete execution procedure, from PDF input to refined Markdown output.

### Step 1 — Figure and Table Detection (`doclayout.py`)

Activate the document layout environment:

```bash
conda activate env_doclayout
```

Place the input PDF papers in the `data/` directory.

Run:

```bash
python doclayout.py
```

By default, the script processes all PDF files in `data/`.

Alternatively, process an individual PDF:

```bash
python doclayout.py --pdf-path data/paper_name.pdf
```

---

### Step 2 — OCR-Based Figure Filtering (`ocr_filter.py`)

Remain in `env_doclayout` and run:

```bash
python ocr_filter.py
```

This script automatically scans the `result/` directory and processes the extracted figures and tables. A figure is retained only if its recognized text contains at least one keyword from each of the two keyword groups.

---

### Step 3 — Batch Figure Separation (`sep.sh (sep.py)`)

`sep.sh` processes every paper directory under `result_filtered/` in one batch. It calls `sep.py`, which uses a pretrained detection model to locate and crop individual subfigures within each extracted figure.

Run from the repository root:

```bash
bash sep.sh
```

For each paper directory, the script:

1. Checks that `extracted_figures/` exists; directories without it are skipped with a warning.
2. Creates a `separated_subfigures/` output folder.
3. Runs `sep.py` with `--img` set to the paper's `extracted_figures/` and `--output_dir` set to its `separated_subfigures/`.

Output is organized per paper, with each detected subfigure saved as a separate image. Useful options include `--thresh` (detection threshold), `--nms_iou` (NMS suppression threshold), `--expand_pixels` / `--expand_ratio` (bounding-box expansion), and `--no_expand` (disable expansion).

---

### Step 4 — Chart Classification and OCR Filtering (`classifier_filter.py`)

Activate the classification environment:

```bash
conda activate env_swin
```

Run:

```bash
python classifier_filter.py
```

This stage uses a pretrained Swin Transformer model to classify the separated subfigures and applies a second round of OCR-based filtering. Each subfigure is classified into one of the following chart categories:

- `horizontal_bar`
- `vertical_bar`
- `line`
- `scatter`
- `scatter-line`

Only images assigned to the supported chart categories with sufficient confidence are retained.

A chart is retained if its recognized text contains at least one keyword from either keyword group. Charts containing none of these keywords are excluded. The output JSON records the filenames and predicted categories of the retained charts, for example::

```json
[
    {
        "filename": "figure_1_1_0.92.png",
        "class": "line"
    },
    {
        "filename": "figure_2_1_0.89.png",
        "class": "vertical_bar"
    }
]
```

---

### Step 5 — Numerical Data Extraction (`api.py`)

Activate the API processing environment:

```bash
conda activate env_doclayout
```

Navigate to `result_filtered/`:

```bash
cd result_filtered
```

Set up API configuration and then run the extraction script for one or multiple papers:

```bash
python ../api.py paper_name1 paper_name2 --api-model gpt-4o
```

The API uses multimodal LLMs to reconstruct numerical data from the classified chart images.

**Recovery Mode**

Failed images can be reprocessed using:

```bash
python ../api.py paper_name1 --recover
```

Recovery mode reads `error.json` and retries the corresponding images.

---

### Step 6 — Data Refinement (`api_refine.py`)

Remain in `result_filtered/` and activate `env_doclayout` .

Set up API configuration and then run:

```bash
python ../api_refine.py paper_name1 paper_name2 --swin-model {your_path} --api-model gpt-4o
```

This stage further improves the extracted Markdown results by incorporating figure captions and additional chart information from the original PDF papers.

---

## 📊 Final Output

The complete pipeline produces structured Markdown reports containing reconstructed numerical data from scientific chart images.

The final outputs include:

| Output | Description |
|---|---|
| `figure_text_E.json` | OCR results for extracted figures. |
| `table_text_E.json` | OCR results for extracted tables. |
| `classification_chart.json` | Chart filenames and predicted chart categories after classification and OCR filtering. |
| `chart_data_extraction_report_*.md` | Numerical data extracted from classified charts. |
| `error.json` | Failed image records for recovery processing. |
| `optimized_report.md` | Refined Markdown report incorporating figure captions and chart information. |

---

## 📈 Benchmark

To verify why **locating and separating** figures matters, see our [ChartDensity-Bench](https://github.com/issair/ChartDensity-Bench). This benchmark systematically evaluates how visual density affects MLLM-based numerical reconstruction by composing the same source charts into compound figures.

For full benchmark construction details, evaluation metrics (RFV, CAR, PR, ACR, EC, MAE/RMSE), and model comparisons, see the paper: [ChartDensity-Bench on arXiv](https://arxiv.org/abs/2609.38781)

---

## 📜 License

### SciFig2Data Code

The original code developed for **SciFig2Data** is intended to be released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.

### Third-Party Models and Dependencies

SciFig2Data integrates the following open-source projects and their pretrained models:

- **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)** — used for document layout analysis and figure/table detection. The upstream project is distributed under the **GNU AGPL-3.0 License**.
- **[figure-separator](https://github.com/apple2373/figure-separator)** — used for compound figure decomposition and subfigure detection. Please refer to the original repository for applicable licensing terms.
- **[Swin-chart](https://github.com/moured/Swin-chart)** — used for chart image classification. Please refer to the original repository for applicable licensing terms.

The original implementations, pretrained weights, and other third-party materials remain subject to their respective licenses and terms. They are not relicensed under the SciFig2Data license.

Users are responsible for complying with the applicable licenses and terms when using, modifying, or redistributing these third-party components and model weights.

For academic use, please also cite the corresponding publications listed below.

---

## 📚 Citation

If you find **SciFig2Data** useful in your research, please consider citing our paper and the following works that provide the core models used in this framework.

### SciFig2Data

```bibtex
% Our paper:
@misc{wu2026scifig2data,
      title={SciFig2Data: A Workflow for Recovering Structured Data from Scientific Figures}, 
      author={Xinhe Wu and Yicheng Ye and Yadong Jin},
      year={2026},
      eprint={2611.31787},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2611.31787}, 
}
```

### Third-Party Models

```bibtex
% DocLayout-YOLO
@misc{zhao2024doclayoutyolo,
    title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception},
    author={Zhao, Zhiyuan and Kang, Hengrui and Wang, Bin and He, Conghui},
    year={2024},
    eprint={2410.12628},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2410.12628}
}

% Figure Separator
@inproceedings{tsutsui2017compound,
  title     = {A Data Driven Approach for Compound Figure
               Separation Using Convolutional Neural Networks},
  author    = {Tsutsui, Satoshi and Crandall, David},
  booktitle = {2017 14th IAPR International Conference on
               Document Analysis and Recognition (ICDAR)},
  year      = {2017},
  doi       = {10.1109/ICDAR.2017.93}
}

% Swin-Chart
@article{dhote2024swinchart,
  title   = {Swin-chart: An efficient approach for chart classification},
  author  = {Dhote, Anurag and Javed, Mohammed and
             Doermann, David S.},
  journal = {Pattern Recognition Letters},
  volume  = {185},
  pages   = {203--209},
  year    = {2024},
  doi     = {10.1016/j.patrec.2024.08.012}
}
```