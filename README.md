# HW3 Principal Component Analysis - Demo

This repository contains `hw3.py`, which implements PCA helper functions used for an assignment.

What I added
- A runnable demo in `hw3.py` (an `if __name__ == "__main__"` block):
  - If a `.npy` or `.npz` file is present in the project folder, it will be loaded and used.
  - If no dataset is found, a synthetic dataset is generated for a smoke test.
  - The demo computes covariance, top eigenvectors, reconstructs the first sample, saves `reconstruction_demo.png`, and attempts to show the plot.
- `requirements.txt` listing the required packages.

Quick start (PowerShell)

```powershell
# create and activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# run the demo
python hw3.py
```

Notes
- `load_and_center_dataset` expects a NumPy `.npy` or `.npz` file. If your data is in another format (CSV, images), please convert or tell me and I can add a loader.
- `display_image` expects arrays that reshape to `(218, 178, 3)` (high-res color) and `(60, 50)` (small grayscale). The demo creates synthetic data matching the small image size.
- If you want me to load a specific dataset that's on your machine, either place the `.npy`/`.npz` file in the project folder or tell me its path and I'll run the demo with it.
