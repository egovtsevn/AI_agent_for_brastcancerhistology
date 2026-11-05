### AI-Assisted Analysis of Breast Cancer Histopathology Images

## Overview

It is an AI-assisted tool for the quantitative analysis of breast cancer histopathology whole-slide images (WSIs). It combines deep learning-based image feature extraction, machine learning classification, tissue composition analysis, and literature retrieval to support the preparation of structured pathology reports.

The project is designed to analyze selected regions of breast biopsy slides, estimate the relative proportions of different tissue components, and generate a draft report based on the College of American Pathologists (CAP) breast reporting framework.

## Key Features

- **Whole-slide image analysis:** Processing of histopathological images in TIFF format using OpenSlide.
- **Deep feature extraction:** Extraction of image patch embeddings using a pretrained feature extraction model.
- **Tissue classification:** Classification of image patches using a machine learning classifier.
- **Tissue composition analysis:** Pixel-level aggregation of overlapping patch predictions to estimate the relative areas of tissue components.
- **Large-slide processing:** Chunk-based processing of large image regions to reduce memory requirements.
- **Literature retrieval:** Automated PubMed searches for relevant scientific publications using the NCBI Entrez API.
- **CAP-oriented report generation:** Preparation of a structured breast biopsy report with image-derived measurements and manually supplied clinical information.

## Analysis Pipeline

The main workflow consists of the following steps:

1. **Input:** A whole-slide image (WSI) in TIFF format and the coordinates and dimensions of the region to be analyzed.
2. **Patch generation:** The selected image region is divided into overlapping patches.
3. **Feature extraction:** A pretrained deep learning model extracts feature embeddings from each patch.
4. **Classification:** A machine learning classifier assigns a tissue category to each patch.
5. **Spatial aggregation:** Predictions from overlapping patches are combined using pixel-level hard voting to obtain a tissue classification mask.
6. **Quantification:** The relative area occupied by each tissue category is calculated.
7. **Literature retrieval:** Relevant publications are retrieved from PubMed based on the tumor type and available clinical information.
8. **Report generation:** Image-derived measurements and manually provided clinical information are combined into a structured CAP-oriented report.

## Tissue Categories

The image classifier distinguishes four broad morphological categories:

| Class | Description |
|---|---|
| Invasive epithelium | Tissue regions showing morphological patterns consistent with invasive carcinoma |
| Non-invasive epithelium | Epithelial tissue with a non-infiltrative growth pattern |
| Necrosis | Necrotic tissue |
| Other | Heterogeneous category that may include stroma, adipose tissue, inflammatory infiltrate, blood, and other structures |

The non-invasive epithelium category does not distinguish benign epithelium from in-situ carcinoma. Therefore, its estimated proportion should not be interpreted as the extent of ductal carcinoma in situ (DCIS).

## Report Generation

The reporting module uses a CAP-oriented breast template.

The report combines:
- Quantitative tissue composition measurements obtained from image analysis.
- Clinical and pathological information supplied manually.
- A literature-based comment generated using retrieved PubMed publications.

## Technologies

- Python
- PyTorch
- NumPy
- OpenSlide
- Pillow
- scikit-learn-compatible classification interface
- Biopython (NCBI Entrez)
- Groq API

## Limitations

- The classifier provides broad morphological categories and cannot establish a definitive histopathological diagnosis.
- Non-invasive epithelium includes both benign and neoplastic structures.
- The heterogeneous "Other" category cannot be interpreted as a specific tissue component.
- Tissue percentages describe the analyzed image region and should not automatically be extrapolated to the entire biopsy specimen.
- Clinical parameters such as histological grade, tumor size, lymphovascular invasion, and biomarker status require independent assessment.
- Literature retrieval provides scientific context and does not establish clinical associations by itself.
- Generated reports must be reviewed and verified by a qualified pathologist before clinical use.
