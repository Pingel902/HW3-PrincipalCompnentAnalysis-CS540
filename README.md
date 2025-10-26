# Principal Component Analysis (PCA) Image Processing

This project implements Principal Component Analysis (PCA) for image processing and dimensionality reduction. It provides functionality to analyze, reconstruct, and perturb images using eigenvalue decomposition and covariance matrices.

## Features

- Load and center image datasets
- Compute covariance matrices
- Calculate eigenvalues and eigenvectors
- Project and reconstruct images using PCA
- Perturb images with controlled noise
- Visualize original and reconstructed images

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Quick Start (PowerShell)

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

## Functions

### `load_and_center_dataset(filename)`
Loads a dataset from a NumPy file and centers it by subtracting the mean.

### `get_covariance(dataset)`
Computes the covariance matrix for the centered dataset.

### `get_eig(S, k)`
Calculates the top k eigenvalues and corresponding eigenvectors of the covariance matrix.

### `get_eig_prop(S, prop)`
Finds eigenvalues and eigenvectors that explain a given proportion of variance.

### `project_and_reconstruct_image(image, U)`
Projects an image onto the eigenvectors and reconstructs it.

### `display_image(im_orig_fullres, im_orig, im_reconstructed)`
Displays the original high-resolution image, original image, and reconstructed image side by side.

### `perturb_image(image, U, sigma)`
Adds Gaussian noise to the image projection with a specified standard deviation.

## Usage

The script can be run directly and will:
1. Look for `.npy` or `.npz` files in the current directory
2. Load and process the image data
3. Perform PCA
4. Display and save the reconstruction results

If no dataset is found, it will create a synthetic demo dataset.

## Output

The script generates a visualization comparing:
- Original high-resolution image (218 x 178 x 3)
- Original image (60 x 50)
- Reconstructed image after PCA

The visualization is saved as 'reconstruction_demo.png' in the working directory.

## Implementation Details

- Uses SciPy's `eigh` function for eigenvalue decomposition
- Implements PCA using covariance matrix method
- Supports both single-channel and RGB images
- Handles various input data formats (.npy, .npz)
- Includes proper error handling and data validation

## Notes

- `load_and_center_dataset` expects a NumPy `.npy` or `.npz` file. If your data is in another format (CSV, images), you'll need to convert it first.
- `display_image` expects arrays that reshape to `(218, 178, 3)` (high-res color) and `(60, 50)` (small grayscale).
- To use a specific dataset, place the `.npy`/`.npz` file in the project folder or provide its full path.
