import sys
import tensorflow as tf
import numpy as np
import os, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def normalize_minus1_1(arr):
    """Normalize an array to the range [-1, 1]."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.float32)

    return 2 * (arr - arr_min) / (arr_max - arr_min) - 1
def pad_to_32(cube):
    if cube.shape == (31, 31, 31):
        return np.pad(cube, pad_width=((0, 1), (0, 1), (0, 1)), mode="constant")
    elif cube.shape == (32, 32, 32):
        return cube
    else:
        raise ValueError(f"Unexpected shape: {cube.shape}")

def serialize_example(flat_array, index):
    return tf.train.Example(features=tf.train.Features(feature={
        "data": tf.train.Feature(float_list=tf.train.FloatList(value=flat_array)),  # Change to FloatList
        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
    })).SerializeToString()

def create_tfrecords_from_npy(output_dir, num_threads=8, batch_size=1000):
    file = '/home/leonardo/testing_saxs/test_data_cont/pdbs/many_dark_norm_voxels_1_0/many_dark.txt'
    print(f"Loading all .npy arrays from: {file}")

    # Read the list of file paths from many_dark.txt
    with open(file, 'r') as f:
        npy_files = [line.strip() for line in f if line.strip()]

    print(f"Found {len(npy_files)} .npy files. Loading and padding arrays...")
    print("Loading and padding arrays...")
    arrays = [pad_to_32(np.load(f)).astype(np.float32) for f in tqdm(npy_files, desc="Padding arrays")]
    arrays = np.array([np.clip(i, 0, 1) for i in np.array(arrays[:-1])])

    # Writers
    train_writer = tf.io.TFRecordWriter(os.path.join(output_dir, "train.tfrecords"))
    test_writer = tf.io.TFRecordWriter(os.path.join(output_dir, "test.tfrecords"))

    print(f"Calculating differences...")


    def process_pair(i, j, idx):
        diff = arrays[i] - arrays[j]
        normalize_minus1_1(diff)
        if diff.sum() == 0:
            return None  # skip identical
        flat = diff.flatten().astype(np.float32)  # Change to float32
        example = serialize_example(flat, idx)
        return (idx, example)

    idx = 0
    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(arrays) * (len(arrays) - 1) // 2, desc="Pairwise diffs") as pbar:
            for i in range(len(arrays)):
                for j in range(i + 1, len(arrays)):
                    futures.append(executor.submit(process_pair, i, j, idx))
                    idx += 1

                    # process in batches to avoid memory blow-up
                    if len(futures) >= batch_size:
                        for fut in as_completed(futures):
                            result = fut.result()
                            if result is not None:
                                idx_out, ex = result
                                if idx_out % 5 == 0:
                                    test_writer.write(ex)
                                else:
                                    train_writer.write(ex)
                        futures = []
                        pbar.update(batch_size)

            # process leftovers
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    idx_out, ex = result
                    if idx_out % 5 == 0:
                        test_writer.write(ex)
                    else:
                        train_writer.write(ex)
                pbar.update(1)

    train_writer.close()
    test_writer.close()

    print(f"✅ Done: TFRecords written to {output_dir}")

# ───────────────────────────
if __name__ == "__main__":
    create_tfrecords_from_npy(
        output_dir=sys.argv[1],
        num_threads=8,
        batch_size=300  # tweak based on memory
    )
