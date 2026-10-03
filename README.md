# CopyCatch.io 🔍

**CopyCatch.io** is a modular, high-performance local image duplicate detector designed to swiftly identify duplicate and near-duplicate image clusters. By combining sub-millisecond Hamming distance lookups with a secondary machine learning fallback, CopyCatch delivers rapid, accurate clustering without automated file deletion—putting full control in the user's hands.

---

## 🌟 Key Features

- **Two-Tier Detection Pipeline**:
  - **Fast Path (Direct Match)**: Calculates 64-bit Perceptual Hashes ($pHash$) using Discrete Cosine Transform (DCT) and performs vector indexing via **FAISS** for Hamming distances $\le 5$.
  - **ML Fallback (Ambiguous Match)**: For borderline cases (Hamming distance $6\le d \le 15$), utilizes a quantized **MobileNetV3 ONNX model** to extract feature embeddings and verify similarity via Cosine Distance ($\ge 0.85$).
- **Strict Duplicate Filtering**: Automatically ignores unique (non-duplicate) images to keep the review interface clean and actionable.
- **Graph-Based Clustering**: Uses Connected Components (via **NetworkX**) to group duplicate relationships into intuitive visual clusters.
- **Privacy First & Local Execution**: Fully offline execution—no images or metrics are sent to external servers.
- **Interactive UI**: Built with Streamlit, featuring native OS folder pickers (`tkinter`) and instant in-memory thumbnail previews.

---

## 🛠️ Tech Stack

- **Frontend & UI**: Streamlit, Tkinter (native OS file dialogs)
- **Hashing & Vector Search**: Scipy (DCT), FAISS (`faiss-cpu` / `IndexBinaryFlat`)
- **Machine Learning**: ONNX Runtime (`onnxruntime`), MobileNetV3 (quantized embedding model)
- **Graph & Clustering**: NetworkX
- **Image Processing**: Pillow (PIL), NumPy

---

## 📐 Project Architecture

```
CopyCatch.io/
├── app.py                     # Application entry point
├── config/
│   └── settings.py            # Global thresholds, model paths, & configs
├── core/
│   ├── hashing/
│   │   ├── base.py            # Abstract hashing interface
│   │   ├── phash.py           # 64-bit DCT-based Perceptual Hasher
│   │   └── dhash.py           # Difference Hasher (alternative)
│   ├── indexer/
│   │   └── faiss_index.py     # Binary FAISS index wrapper (IndexBinaryFlat)
│   ├── ml/
│   │   └── onnx_verifier.py   # MobileNetV3 ONNX embedding extractor
│   ├── grouping/
│   │   └── graph_cluster.py   # NetworkX Connected Components clustering
│   ├── image_loader.py        # Directory scanner & thumbnail generator
│   └── models.py              # Dataclasses (ImageItem, DuplicateGroup)
├── pipeline/
│   └── detector.py            # Main duplicate detection coordinator
└── ui/
    ├── components/
    │   ├── folder_picker.py   # Native OS folder browser
    │   ├── group_card.py      # Duplicate group renderer
    │   └── stats_bar.py       # Metrics summary widget
    └── views/
        └── main_layout.py     # Streamlit view coordinator
```

---

## ⚡ How It Works

1. **Scan Directory**: Scans the targeted local folder for supported image formats (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
2. **Compute Hashes**: Generates 64-bit $pHash$ representations for each image.
3. **FAISS Range Search**: Populates a `faiss.IndexBinaryFlat` vector index and evaluates bitwise Hamming distances across all pairs:
   - $d \le 5 \implies$ **Direct Duplicate**
   - $5 < d \le 15 \implies$ **Ambiguous Match** (sent to ONNX fallback)
4. **ONNX Feature Extraction**: Evaluates cosine similarity of MobileNetV3 embeddings for ambiguous pairs. Matches with similarity $\ge 0.85$ are accepted as duplicates.
5. **Graph Clustering**: Builds an adjacency graph where images are nodes and duplicate relationships are edges. Connected components with $\ge 2$ nodes are rendered as duplicate groups.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
Install all required libraries into your Python environment:

```bash
python -m pip install streamlit faiss-cpu onnxruntime pillow numpy scipy networkx
```

### 3. Run Application
Launch the Streamlit web application:

```bash
python -m streamlit run app.py
```

---

## 📄 License

MIT License. Feel free to use, modify, and extend!