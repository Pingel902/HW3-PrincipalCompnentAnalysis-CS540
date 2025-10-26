from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

def load_and_center_dataset(filename):
    
    data = np.load(filename)
    dataCentered = data - np.mean(data, axis=0)
    return dataCentered.astype(float)


def get_covariance(dataset):
    n,d = dataset.shape
    matrixS =(1.0/ (n-1)) * np.dot(dataset.T, dataset)
    return matrixS

def get_eig(S, k):
    # Your implementation goes here!
    e = S.shape[0]
    vals, vectors = eigh(S, subset_by_index=[e-k, e-1])
    vals = vals[::-1]
    vectors = vectors[:, ::-1]
    diag = np.diag(vals)
    return diag, vectors

def get_eig_prop(S, prop):
    e = S.shape[0]
    vals, vectors = eigh(S)
    valsSorted = vals[::-1]
    vectorsSorted = vectors[:, ::-1]
    total = np.sum(valsSorted)
    sum = 0
    for i in range(e):
        sum += valsSorted[i]
        if sum/total >= prop :
            sum = i + 1
            break
    topVals = valsSorted[:sum]
    topVectors = vectorsSorted[:,:sum]
    diagonal = np.diag(topVals)
    return diagonal, topVectors

def project_and_reconstruct_image(image, U):
    # Your implementation goes here!
    weights = np.dot(U.T, image)
    newImage = np.dot(U, weights)
    return newImage

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    im_orig_fullres = im_orig_fullres.reshape(218, 178, 3)
    ax1.imshow(im_orig_fullres.astype(np.uint8))
    ax1.set_title("Original High Res")
    
    ax2.imshow(im_orig.reshape(60, 50), aspect='equal')
    ax2.set_title("Original")
    fig.colorbar(ax2.images[0], ax=ax2)
    
    # Display reconstructed image
    ax3.imshow(im_reconstructed.reshape(60, 50), aspect='equal')
    ax3.set_title("Reconstructed")
    fig.colorbar(ax3.images[0], ax=ax3)
    
    return fig, ax1, ax2, ax3


    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    oldWeights = np.dot(U.T, image)
    noise = np.random.normal(0, sigma, oldWeights.shape)
    newWeights = oldWeights + noise
    newImage = np.dot(U, newWeights)
    return newImage


def _try_load_numpy_file(filepath):
    """Load a .npy or .npz file and return a 2D array suitable for this assignment.

    If a .npz with multiple arrays is provided, pick the first array found.
    """
    arr = np.load(filepath, allow_pickle=True)
    # .npz returns an NpzFile which acts like a dict
    if hasattr(arr, 'files'):
        # take the first array in the archive
        if len(arr.files) == 0:
            raise ValueError(f"No arrays found inside {filepath}")
        key = arr.files[0]
        data = arr[key]
    else:
        data = arr
    data = np.asarray(data)
    # If data is 1D and looks like a single flattened image, make it 2D with one sample
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # If data is 3D (images), flatten per-sample
    if data.ndim == 3:
        n = data.shape[0]
        data = data.reshape(n, -1)
    # Ensure shape is (n_samples, n_features)
    if data.ndim != 2:
        raise ValueError(f"Loaded array has unsupported shape {data.shape}")
    return data


if __name__ == "__main__":
    # Demo runner: tries to find a .npy/.npz dataset in the current folder.
    cwd = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    patterns = [os.path.join(cwd, '*.npy'), os.path.join(cwd, '*.npz')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if files:
        datafile = files[0]
        print(f"Loading dataset from: {datafile}")
        try:
            data = _try_load_numpy_file(datafile)
        except Exception as e:
            print(f"Failed to load {datafile}: {e}")
            sys.exit(1)
    else:
        # No dataset found: create a synthetic dataset matching expected shapes
        print("No .npy/.npz dataset found in the folder. Creating a synthetic demo dataset.")
        n_samples = 50
        # small image size used in display_image: 60 x 50 -> 3000 features
        small_h, small_w = 60, 50
        d = small_h * small_w
        rng = np.random.default_rng(0)
        data = rng.normal(loc=128, scale=50, size=(n_samples, d)).astype(float)

        # create a fake high-res color image for display (218 x 178 x 3)
        highres = rng.integers(0, 256, size=(218 * 178 * 3,), dtype=np.uint8)

    # Center dataset
    data_centered = data - np.mean(data, axis=0)

    # Compute covariance and eigendecomposition
    S = get_covariance(data_centered)
    k = min(20, S.shape[0])
    diag_vals, U = get_eig(S, k)

    # Project and reconstruct the first sample
    sample = data_centered[0]
    reconstructed = project_and_reconstruct_image(sample, U)

    # If we created a synthetic highres, use it; otherwise try to build a dummy highres from sample
    try:
        im_full = highres
    except NameError:
        # expand the small image to a fake color high-res by tiling and stacking
        small_img = (data[0]).reshape(small_h, small_w)
        im_full = np.tile(small_img.repeat(3, axis=0).repeat(3, axis=1)[:218, :178], (3,))

    # display and save
    try:
        fig, ax1, ax2, ax3 = display_image(im_full, data[0], reconstructed)
        outname = os.path.join(cwd, 'reconstruction_demo.png')
        fig.savefig(outname)
        print(f"Saved reconstruction image to: {outname}")
        plt.show()
    except Exception as e:
        print(f"Display failed: {e}\nSaved only numeric results.")


