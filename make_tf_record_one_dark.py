import sys
import tensorflow as tf
import numpy as np
import os, glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize the input data to a range between 0 and 1.

    Parameters
    ----------
    data : np.ndarray
        The input data (array) to be normalized.

    Returns
    -------
    np.ndarray
        The normalized data.
    """
    # Ensure the data is a NumPy array
    data = np.asarray(data)

    # Compute the min and max of the data
    min_val = np.min(data)
    max_val = np.max(data)

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return np.zeros_like(data)  # Return an array of zeros if all values are the same

    # Normalize to [0, 1] range
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data

def pad_to_32(cube):
    if cube.shape == (31, 31, 31):
        return np.pad(cube, pad_width=((0, 1), (0, 1), (0, 1)), mode='constant')
    elif cube.shape == (32, 32, 32):
        return cube
    else:
        raise ValueError(f"Unexpected shape: {cube.shape}")


def load_all_npy_files(directory):
    files = glob.glob(os.path.join(directory, "*.npy"))
    if not files:
        raise ValueError(f"No .npy files found in directory: {directory}")
    return files


def write_bead(voxel_grid, threshold=0.5):
    # Binary: 1 if voxel >= threshold, else -1
    labeled = np.zeros(voxel_grid.shape, dtype=np.float32)  # Change to float32
    labeled[voxel_grid >= threshold] = 1.0
    labeled[voxel_grid <= -threshold] = -1.0
    return labeled


def serialize_example(flat_array, index):
    return tf.train.Example(features=tf.train.Features(feature={
        "data": tf.train.Feature(float_list=tf.train.FloatList(value=flat_array)),  # Change to FloatList
        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
    })).SerializeToString()


def write_tfrecord(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in tqdm(examples, desc=f"Writing {filename}"):
            writer.write(ex)


def normalize_minus1_1(arr):
    """Normalize an array to the range [-1, 1]."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return 2 * (arr - arr_min) / (arr_max - arr_min) - 1

def standardise(arr):
    """Standardise an array with mean 0 and std 1."""
    return arr - arr.mean() / arr.std()
def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize the input data to a range between 0 and 1.

    Parameters
    ----------
    data : np.ndarray
        The input data (array) to be normalized.

    Returns
    -------
    np.ndarray
        The normalized data.
    """
    # Ensure the data is a NumPy array
    data = np.asarray(data)

    # Compute the min and max of the data
    min_val = np.min(data)
    max_val = np.max(data)

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return np.zeros_like(data)  # Return an array of zeros if all values are the same

    # Normalize to [0, 1] range
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data
def create_tfrecords_from_npy(directory, reference, output_dir, num_threads=8,):
    print(f"Loading all .npy arrays from: {directory}")
    print(f"Loading reference .npy arrays from: {reference}")
    npy_files = load_all_npy_files(directory)

    print("Loading and padding arrays...")
    arrays = [pad_to_32(np.load(f)).astype(np.float32) for f in tqdm(npy_files, desc="Padding arrays")]
    ref = pad_to_32(np.load(reference)).astype(np.float32)  # Ensure ref is float32
    arrays.append(ref)
    print("Processing arrays to binary beads...")
    # beads = [write_bead(arr) for arr in tqdm(arrays, desc="Creating binary beads")]

    print(f"Calculating differences...")
    ref = np.array([arrays[-1]])
    ref = np.clip(ref,0,1)
    arrays = np.array([np.clip(i,0,1) for i in np.array(arrays[:-1])])
    differences = arrays - ref

    # Normalize each difference to [-1, 1] range
    differences = np.array([normalize_minus1_1(diff)
                            for diff in tqdm(differences, desc="Normalizing differences")
                            if abs(diff).sum() != 0])

    print(f"Total number of samples: {len(differences)}")

    examples_train = []
    examples_test = []

    def process_sample(idx):
        flat = differences[idx].flatten().astype(np.float32)
        return idx, serialize_example(flat, idx)

    print("Serializing examples using multithreading...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_sample, range(len(differences))),
                            total=len(differences), desc="Serializing TFRecords"))

    for idx, ex in results:
        if idx % 5 == 0:
            examples_test.append(ex)
        else:
            examples_train.append(ex)

    print("Writing TFRecords...")
    os.makedirs(output_dir, exist_ok=True)
    write_tfrecord(os.path.join(output_dir, "train.tfrecords"), examples_train)
    write_tfrecord(os.path.join(output_dir, "test.tfrecords"), examples_test)

    print(f"✅ Done: {len(examples_train)} training and {len(examples_test)} test samples.")


# ───────────────────────────
# ───────────────────────────
if __name__ == "__main__":
    create_tfrecords_from_npy(
        directory=sys.argv[1],
        reference=sys.argv[2],
        output_dir=sys.argv[3],
        num_threads=8
    )
