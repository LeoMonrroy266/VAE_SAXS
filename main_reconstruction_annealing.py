# coding:utf-8
import numpy as np
import tensorflow as tf
from functools import partial
from scipy.optimize import dual_annealing
import argparse
import os
import multiprocessing
import map2iq_shape as map2iq
import pdb2voxel_scaled as pdb2voxel
from VAE import VAE
from data_class import ScatterData
from density_manipulation import create_ccp4_map_

def conditional_addition_to_b(a, b):
    b_added = b.copy()
    mask1 = (a == -1) & (b == 1)
    mask2 = (a == 1) & (b == 0)
    b_added[mask1] = (a + b)[mask1]
    b_added[mask2] = (a + b)[mask2]
    return b_added

def compute_score(latent, vae, ground_voxel, s_factor, saxs_data):
    latent = latent.reshape(1, -1).astype(np.float32)
    voxel = vae.decode(latent).numpy()[0, :31, :31, :31, 0]
    extrapolated = conditional_addition_to_b(voxel, ground_voxel).clip(0, 1)

    # Use multiprocessing to score in parallel (here only 1 candidate, still can use Pool.map for batch later)
    pool = multiprocessing.Pool(processes=min(20, os.cpu_count()))
    func = partial(map2iq.run, iq_file=saxs_data, ground_state_voxel=ground_voxel, ground_state_scale=s_factor)
    score = pool.map(func, [(extrapolated, s_factor)])
    pool.close()
    pool.join()
    return -score[0]

def annealing_callback(x, f, context):
    print(f"Iteration {context['nfev']}: Current score = {-f:.4f}")
    return False  # return True to stop early

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iq_path', required=True)
    parser.add_argument('--pdb', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    group_init_parameter = np.loadtxt(os.path.join(args.model_path, 'latent_init.txt'))

    encoder = tf.keras.layers.TFSMLayer(f"{args.model_path}/encoder_model", call_endpoint="serving_default")
    decoder = tf.keras.layers.TFSMLayer(f"{args.model_path}/decoder_model", call_endpoint="serving_default")
    vae = VAE(256, encoder=encoder, decoder=decoder)

    pdb2voxel.main(args.pdb, 'ground_state', args.output_folder)
    ground_voxel = np.load(f'{args.output_folder}/ground_state_continuous_cube.npy')[:31,:31,:31]
    s_factor = np.load(os.path.join(args.output_folder, 'ground_state_scale_factor.npy'))
    saxs_data = ScatterData(args.iq_path, ',')

    bounds = [(mu - 3*sigma, mu + 3*sigma) for mu, sigma in group_init_parameter]

    result = dual_annealing(
        func=partial(compute_score, vae=vae, ground_voxel=ground_voxel, s_factor=s_factor, saxs_data=saxs_data),
        bounds=bounds,
        maxiter=80,
        callback=annealing_callback
    )

    print('R²', -result.fun)
    best_voxel = vae.decode(result.x.reshape(1, -1)).numpy()[0, :31, :31, :31, 0]
    create_ccp4_map_(best_voxel, f"{args.output_folder}/best_diff.ccp4", voxel_size=3.33,
                     scaling_factor_file=os.path.join(args.output_folder, 'ground_state_scale_factor.npy'))
