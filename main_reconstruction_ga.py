# coding:utf-8
import numpy as np
from numpy.ma.core import concatenate

import map2iq_main as map2iq
import time
import multiprocessing
import region_search
import os
import pdb2voxel_scaled as pdb2voxel
import argparse
import processSaxs as ps
from functools import partial
import tensorflow as tf
import os
import logging
import absl.logging
import align2PDB as align2pdb
from VAE import *
from density_manipulation import *
from data_class import ScatterData


class Evolution:
    def __init__(self, output_folder,pdb_scale, voxel_pdb, process_result, vae, max_iter=80):
        self.output_folder = output_folder
        self.k_cube = 10  # e.g., ±3σ
        self.latent_mu = group_init_parameter[:, 0]
        self.latent_sigma = group_init_parameter[:, 1]
        self.latent_lower = self.latent_mu - self.k_cube * self.latent_sigma
        self.latent_upper = self.latent_mu + self.k_cube * self.latent_sigma

        self.iteration_step = 0
        self.counter = 0
        self.max_iter = max_iter
        self.process_result = process_result
        self.vae = vae
        self.voxel_pdb = voxel_pdb
        self.voxel_pdb_scale = pdb_scale
        # length of latent vector
        self.gene_length = 256
        # numbers of two-point crossing one time.
        self.exchange_gene_num = 100
        # initial population
        self.population = 500
        # population after cycle
        self.inheritance_num = 500
        # initial scale factor should match the number of latent vectors, aka population
        self.s = (np.ones(self.population) * self.voxel_pdb_scale).reshape(1, -1)
        # every iteration step, keep top 10 samples unchanged
        self.remain_best_num = 10
        # used for averaging when getting the final result
        self.statistics_num = 10
        self.compute_score = map2iq.run
        self.group = self.generate_original_group(self.population)
        self.group = np.hstack((self.group,self.s.T))
        self.scale = (np.ones(self.population)).reshape(1, -1)
        self.group = np.hstack((self.group, self.scale.T))
        self.group_score = self.compute_group_score(self.group)
        self.group, self.group_score = self.rank_group(self.group, self.group_score)

        self.best_so_far = np.copy(self.group[:self.remain_best_num])
        self.best_so_far_score = np.copy(self.group_score[:self.remain_best_num])
        self.score_mat = np.copy(self.group_score[:self.statistics_num]).reshape((1, self.statistics_num))
        self.gene_data = np.copy(self.group[:self.statistics_num]).reshape((1, self.statistics_num, self.gene_length+2))

        print('original input , top5:', self.group_score[:5])
        print('best_so_far, top5:', self.best_so_far_score[:5])
        print('mean_score is:', np.mean(self.group_score))
        print('initialized')


    def run_decode(self, data):
        # batch through the decoder
        rec = self.vae.decode(data.astype(np.float32))
        return rec


    # get scores of all the group based on fitness Function.
    def compute_group_score(self, group):
        latent = group[:, :256]
        s_factor = group[:, -2]

        diff_voxels = self.run_decode(latent)
        diff_voxels = diff_voxels[:, :31, :31, :31, 0].numpy()
        print('Average correlation between voxels:',average_voxel_correlation(diff_voxels))
        print("Latent std per genome:", np.std(latent, axis=1)[:5])

        extrapolated_states = np.array([conditional_addition_to_b(voxel, self.voxel_pdb) for voxel in diff_voxels])

        pool = multiprocessing.Pool(processes=20)
        func = partial(self.compute_score, iq_file=self.process_result,
                       ground_state_voxel=self.voxel_pdb,
                       ground_state_scale=self.voxel_pdb_scale)

        data_input = [(i, j) for i, j in zip(extrapolated_states, s_factor)]

        result = pool.map(func, data_input)
        pool.close()
        pool.join()

        return np.array(result)

    # rank whole group based on their scores.
    def rank_group(self, group, group_score):
        s_factor = group_score.T[1]
        scales = group_score.T[2]
        group_score = group_score.T[0]
        index = np.argsort(group_score)[::-1]
        group = group[index]
        group_score = group_score[index]
        group[:,-2] = s_factor[index]
        group[:, -1] = scales[index]
        return group, group_score

    # two-point crossing
    def exchange_gene(self, selective_gene):
        np.random.shuffle(selective_gene)
        for ii in range(0, self.inheritance_num - self.remain_best_num, 2):
            cross_point = np.random.randint(0, self.gene_length, size=(2 * self.exchange_gene_num))
            cross_point = np.sort(cross_point)
            for jj in range(self.exchange_gene_num):
                random_data = np.random.uniform(low=0, high=1)
                if random_data < 0.8:
                    a = cross_point[jj * 2]
                    b = cross_point[jj * 2 + 1]
                    temp = np.copy(selective_gene[ii, a:b])
                    selective_gene[ii, a:b] = selective_gene[ii + 1, a:b]
                    selective_gene[ii + 1, a:b] = np.copy(temp)

    # mutation operator
    def gene_variation(self, selective_gene):
        # Loop through each individual in the population (excluding the best ones)
        for ii in range(self.inheritance_num - self.remain_best_num):

            # Loop through each gene in the individual
            for jj in range(self.gene_length):

                # Apply mutation with a 50% chance
                if np.random.rand() < 0.5:
                    # Add Gaussian noise to the gene
                    selective_gene[ii, jj] += np.random.normal(0, self.latent_sigma[jj] * self.k_cube)

            # Clip the latent vector to stay within the defined bounds (latent cube)
            selective_gene[ii, :self.gene_length] = np.clip(selective_gene[ii, :self.gene_length],
                                                            self.latent_lower, self.latent_upper)

    # selection operator
    def select_group(self):
        selected_group = np.zeros(shape=(self.inheritance_num - self.remain_best_num, self.gene_length+2))
        selected_group_score = np.zeros(shape=(self.inheritance_num - self.remain_best_num))
        for ii in range(self.inheritance_num - self.remain_best_num):
            a = np.random.randint(0, self.population)
            b = np.random.randint(0, self.population)
            random_data = np.random.uniform(low=0, high=1)
            if random_data > 0.1:
                if a < b:
                    selected_group[ii] = np.copy(self.group[a])
                    selected_group_score[ii] = np.copy(self.group_score[a])
                else:
                    selected_group[ii] = np.copy(self.group[b])
                    selected_group_score[ii] = np.copy(self.group_score[b])
            else:
                if a < b:
                    selected_group[ii] = np.copy(self.group[b])
                    selected_group_score[ii] = np.copy(self.group_score[b])
                else:
                    selected_group[ii] = np.copy(self.group[a])
                    selected_group_score[ii] = np.copy(self.group_score[a])

        self.group = selected_group
        self.group_score = selected_group_score

    def inheritance(self):
        self.select_group()

        # ----- Parameters -----
        elite_repro_fraction = 0.5  # 50% of elites also take part in crossover
        random_fraction = 0.40  # 40% new random genes each generation

        # ----- Select elites to reproduce -----
        n_elite_repro = int(self.remain_best_num * elite_repro_fraction)
        elite_repro_indices = np.random.choice(self.remain_best_num, n_elite_repro, replace=False)
        elites_for_crossover = self.best_so_far[elite_repro_indices]

        # ----- Combine selected population + reproducing elites -----
        crossover_pool = np.concatenate((self.group, elites_for_crossover), axis=0)

        # ----- Apply crossover & mutation -----
        self.exchange_gene(crossover_pool)
        self.gene_variation(crossover_pool)

        # ----- Generate some new random individuals -----
        num_random = int(random_fraction * self.inheritance_num)
        random_genes = np.random.normal(0, 2.0, size=(num_random, self.gene_length + 2))
        for i in range(num_random):
            for j in range(self.gene_length):
                random_genes[i, j] = np.random.normal(group_init_parameter[j, 0],
                                                      group_init_parameter[j, 1])
        random_genes[:, -2] = self.voxel_pdb_scale  # initialize s_factor
        random_genes[:, -1] = 1.0  # initialize scale

        # ----- Build next generation -----
        # Keep the original elites unmodified (strict elitism)
        self.group = np.concatenate((crossover_pool, self.best_so_far, random_genes), axis=0)

        # ----- Evaluate -----
        t1 = time.time()
        self.group_score = self.compute_group_score(self.group)
        t2 = time.time()
        logfile.write('compute_group_score cost:%d\n' % (t2 - t1))

        # ----- Rank & update elites -----
        self.group, self.group_score = self.rank_group(self.group, self.group_score)
        self.gene_data = np.concatenate(
            (self.gene_data, self.group[:self.statistics_num].reshape((1, self.statistics_num, self.gene_length + 2))),
            axis=0
        )
        self.score_mat = np.concatenate(
            (self.score_mat, self.group_score[:self.statistics_num].reshape((1, self.statistics_num))),
            axis=0
        )

        # Update best individuals
        self.best_so_far = np.copy(self.group[:self.remain_best_num])
        self.best_so_far_score = np.copy(self.group_score[:self.remain_best_num])

        # Keep top-N for next generation
        self.group = np.copy(self.group[:self.population])
        self.group_score = np.copy(self.group_score[:self.population])

    # If best sample remains unchanged 15 times, reduce the size of the group.
    # Termination: best unchanged 15 times when group size is 100, or max_iter exceeded.
    def evolution_iteration(self):
        while True:
            t1 = time.time()
            self.inheritance()
            self.iteration_step = self.iteration_step + 1
            t2 = time.time()
            print('iteration_step:', self.iteration_step, 'top5:', self.group_score[:5],
                  '\nmean_score is:%.2f' % np.mean(self.score_mat[-1]), self.population)
            logfile.write('iteration_step_%d' % self.iteration_step)
            logfile.write(' cost:%d \n\n' % (t2 - t1))

            if self.score_mat[-1, 0] < self.score_mat[-2, 0]:
                self.counter = 0
            else:
                self.counter = self.counter + 1
                if self.counter > 10 or self.iteration_step > self.max_iter:
                    self.population = self.population - 100
                    self.inheritance_num = self.inheritance_num - 100
                    self.counter = 0
                    if self.population < 100 or self.iteration_step > self.max_iter:
                        np.savetxt('%s/score_mat.txt' % self.output_folder, self.score_mat, fmt='%.3f')
                        result_sample = self.run_decode(self.group[:self.statistics_num, :self.gene_length])
                        t3 = time.time()
                        gene = self.gene_data.reshape((-1, self.gene_length + 2))
                        voxel_group = self.vae.decode(gene[:,:256])
                        voxel_group = voxel_group.numpy().reshape((-1, self.statistics_num, 32, 32, 32))
                        t4 = time.time()
                        logfile.write('\nvoxel_group cost:%d\n' % (t4 - t3))
                        np.savetxt('%s/bestgene.txt' % output_folder, self.group[0], fmt='%.3f')
                        return result_sample, voxel_group, self.group[:self.statistics_num, -2], self.group[:self.statistics_num, -1], gene[:, -1].reshape((-1, self.statistics_num)), gene[:, -2].reshape((-1, self.statistics_num)),gene[:,:256].reshape((-1, self.statistics_num)),self.group[:self.statistics_num, :self.gene_length]

    def generate_original_group(self, num):
        # Sample uniformly within cube
        original_group = np.random.uniform(self.latent_lower, self.latent_upper, size=(num, self.gene_length))
        return original_group


def average_voxel_correlation(diff_voxels):
    """
    Compute the average pairwise Pearson correlation between all voxel maps

    Parameters
    ----------
    diff_voxels : np.ndarray
        Shape (N, X, Y, Z)

    Returns
    -------
    float
        Mean pairwise correlation between flattened voxel arrays.
    """
    n = diff_voxels.shape[0]
    # Flatten each voxel map into 1D vectors
    voxels_flat = diff_voxels.reshape(n, -1)

    # Compute correlation matrix using dot product
    corr_matrix = np.corrcoef(voxels_flat)

    # Extract upper triangle (without diagonal)
    iu = np.triu_indices(n, k=1)
    avg_corr = np.mean(corr_matrix[iu])

    return avg_corr
def conditional_addition_to_b(a, b):
    """
    Update b based on conditions with a:
    - If a == -1 and b == 1 → b_added = a + b
    - If a ==  1 and b == 0 → b_added = a + b
    - Else keep b unchanged.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays of the same shape.

    Returns
    -------
    b_added : np.ndarray
        New array with updated values.
    """
    b_added = b.copy()

    mask1 = (a == -1) & (b == 1)
    mask2 = (a == 1) & (b == 0)

    b_added[mask1] = (a + b)[mask1]
    b_added[mask2] = (a + b)[mask2]

    return b_added
if __name__ == '__main__':

    BATCH_SIZE = 64
    np.set_printoptions(precision=10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to Keras model/weights used by auto_encoder_t_py3', type=str,
                        required=True)
    parser.add_argument('--iq_path', help='path to experimental data', type=str, required=True)
    parser.add_argument('--output_folder', help='path of output folder where result will be saved', type=str,
                        required=True)
    parser.add_argument('--pdb', help='path to pdb containing starting state', required=True, type=str)
    parser.add_argument('--max_iter', help='maximum number of iteration', default=80, type=int)
    args = parser.parse_args()


    model_path = args.model_path  # Keras weights/model file for auto_encoder_t_py3
    iq_path = args.iq_path
    output_folder = args.output_folder
    pdb = args.pdb
    max_iter = args.max_iter

    os.makedirs(output_folder, exist_ok=True) # Create folder



    # group_init_parameter is well-trained model's distribution of latent vector, used to initialize gene group.
    group_init_parameter = np.loadtxt(os.path.join(model_path, 'latent_init.txt'),delimiter=' ')


    print("Initiating Network...")
    encoder = tf.keras.layers.TFSMLayer(f"{model_path}/encoder_model", call_endpoint="serving_default")
    decoder = tf.keras.layers.TFSMLayer(f"{model_path}/decoder_model", call_endpoint="serving_default")
    vae = VAE(256, encoder, decoder)

    print("Processing Ground state structure...")
    pdb2voxel.main(pdb, 'ground_state', output_folder)
    ground_state_voxel = np.load(f'{output_folder}/ground_state_continuous_cube.npy')
    ground_state_voxel = np.pad(ground_state_voxel, pad_width=((0, 1), (0, 1), (0, 1)), mode="constant")
    ground_state_voxel = np.clip(ground_state_voxel, 0, 1)[:31:,:31,:31]

    pdb_scale = np.load(os.path.join(output_folder, 'ground_state_scale_factor.npy'))
    create_ccp4_map_(ground_state_voxel, f'{output_folder}/ground_state.ccp4', voxel_size=3.33,
                     scaling_factor_file=os.path.join(output_folder, 'ground_state_scale_factor.npy'))
    print("Processing Experimental scattering...")
    saxs_data = ScatterData(iq_path,',')



    logfile = open('%s/log.txt' % output_folder, 'a')

    genetic_object = Evolution(output_folder, voxel_pdb=ground_state_voxel, pdb_scale=pdb_scale, vae=vae, max_iter=max_iter,process_result=saxs_data)
    final_iteration_top_voxels, all_iterations_top_voxels,final_iterations_top_sfactors, final_iterations_top_scales,  all_iterations_top_scales ,all_iterations_top_sfactor, latent_all_iterations,final_iteration_latent_top = genetic_object.evolution_iteration()


    for n,voxel,s_factor, scale,latent in zip(list(range(len(final_iterations_top_sfactors))),final_iteration_top_voxels,final_iterations_top_sfactors,final_iterations_top_scales,final_iteration_latent_top):

        create_ccp4_map_(voxel[:31,:31,:31,0].numpy(), f'{output_folder}/top_{n}_diff.ccp4', voxel_size=3.33, scaling_factor_file=s_factor)
        np.save(f'{output_folder}/top_{n}_diff.npy',voxel[:31,:31,:31,0].numpy())
        np.save(f'{output_folder}/top_{n}_scale.npy', scale)
        np.save(f'{output_folder}/top_{n}_sfactor.npy', s_factor)
        np.save(f'{output_folder}/top_{n}_latent.npy', latent)
        extrapolated = conditional_addition_to_b(voxel[:31,:31,:31,0].numpy(), ground_state_voxel[:31,:31,:31]).clip(0,1)
        np.save(f'{output_folder}/top_{n}_extrapolated.npy', extrapolated)
        create_ccp4_map_(extrapolated, f'{output_folder}/top_{n}_extrapolated.ccp4', voxel_size=3.33,
                         scaling_factor_file=s_factor)

    # Add timers for each step
    logfile.close()
