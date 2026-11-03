# WSI Embedding Extraction

This code extracts feature embeddings from whole-slide images (WSIs) using UNI, a pretrained foundation model developed by Mahmood Lab.

The algorithm sequentially downloads test images from the BEETLE dataset hosted on the Zenodo repository.

Only patches intersecting XML annotations are selected for processing, along with a surrounding area of `grid × grid` patches centered around each annotated patch.

Each WSI is divided into non-overlapping 224 × 224 pixel patches, and an embedding is extracted from each selected patch.

The results are saved as compressed NumPy archives (`.npz`), containing:
- `coords` — spatial coordinates of the extracted patches.
- `embeddings` — corresponding feature embeddings.

Image regions at the slide boundaries that are smaller than 224 pixels in either dimension are excluded from processing.
