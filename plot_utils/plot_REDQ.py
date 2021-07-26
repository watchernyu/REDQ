"""
this one program will be used to basically generate all REDQ related figures for ICLR 2021.
NOTE: currently only works with seaborn 0.8.1 the tsplot function is deprecated in the newer version
the plotting function is originally based on the plot function in OpenAI spinningup
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from redq_plot_helper import *

# the path leading to where the experiment file are located
base_path = '../data/REDQ_ICLR21'

# map experiment name to folder's name
exp2path_main = {
'MBPO-ant': 'MBPO-Ant',
'MBPO-hopper': 'MBPO-Hopper',
'MBPO-walker2d': 'MBPO-Walker2d',
'MBPO-humanoid': 'MBPO-Humanoid',
'REDQ-n10-ant':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd20_ant-v2',
'REDQ-n10-humanoid':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd20_humanoid-v2',
'REDQ-n10-hopper':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd20_hopper-v2',
'REDQ-n10-walker2d':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd20_walker2d-v2',
'SAC-1-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf1_pd20_ant-v2',
'SAC-1-humanoid': 'REDQ_embpo_qmin_piave_n2_m2_uf1_pd20_humanoid-v2',
'SAC-1-hopper': 'REDQ_embpo_qmin_piave_n2_m2_uf1_pd20_hopper-v2',
'SAC-1-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf1_pd20_walker2d-v2',
'SAC-20-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_ant-v2',
'SAC-20-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_walker2d-v2',
'SAC-20-hopper': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_hopper-v2',
'SAC-20-humanoid': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_humanoid-v2',
}
exp2path_N_ablation = {
'REDQ-n15-ant':'REDQ_embpo_qmin_piave_n15_m2_uf20_pd20_ant-v2',
'REDQ-n15-humanoid':'REDQ_embpo_qmin_piave_n15_m2_uf20_pd20_humanoid-v2',
'REDQ-n15-hopper':'REDQ_embpo_qmin_piave_n15_m2_uf20_pd20_hopper-v2',
'REDQ-n15-walker2d':'REDQ_embpo_qmin_piave_n15_m2_uf20_pd20_walker2d-v2',
'REDQ-n5-ant': 'REDQ_embpo_qmin_piave_n5_m2_uf20_pd20_ant-v2',
'REDQ-n5-humanoid': 'REDQ_embpo_qmin_piave_n5_m2_uf20_pd20_humanoid-v2',
'REDQ-n5-hopper': 'REDQ_embpo_qmin_piave_n5_m2_uf20_pd20_hopper-v2',
'REDQ-n5-walker2d': 'REDQ_embpo_qmin_piave_n5_m2_uf20_pd20_walker2d-v2',
'REDQ-n3-ant': 'REDQ_embpo_qmin_piave_n3_m2_uf20_pd20_ant-v2',
'REDQ-n3-humanoid': 'REDQ_embpo_qmin_piave_n3_m2_uf20_pd20_humanoid-v2',
'REDQ-n3-hopper': 'REDQ_embpo_qmin_piave_n3_m2_uf20_pd20_hopper-v2',
'REDQ-n3-walker2d': 'REDQ_embpo_qmin_piave_n3_m2_uf20_pd20_walker2d-v2',
'REDQ-n2-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_ant-v2',
'REDQ-n2-humanoid': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_humanoid-v2',
'REDQ-n2-hopper': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_hopper-v2',
'REDQ-n2-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd20_walker2d-v2',
}
exp2path_redq_variants = {
'REDQ-weighted-hopper': 'REDQ_embpo_qweighted_piave_n10_m2_uf20_pd20_hopper-v2',
'REDQ-minpair-hopper': 'REDQ_embpo_qminpair_piave_n10_m2_uf20_pd20_hopper-v2',
'REDQ-weighted-walker2d': 'REDQ_embpo_qweighted_piave_n10_m2_uf20_pd20_walker2d-v2',
'REDQ-minpair-walker2d': 'REDQ_embpo_qminpair_piave_n10_m2_uf20_pd20_walker2d-v2',
'REDQ-weighted-ant': 'REDQ_embpo_qweighted_piave_n10_m2_uf20_pd20_ant-v2',
'REDQ-minpair-ant': 'REDQ_embpo_qminpair_piave_n10_m2_uf20_pd20_ant-v2',
'REDQ-rem-hopper': 'REDQ_embpo_qrem_piave_n10_m2_uf20_pd20_hopper-v2',
'REDQ-ave-hopper': 'REDQ_embpo_qave_piave_n10_m2_uf20_pd20_hopper-v2',
'REDQ-min-hopper': 'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_hopper-v2',
'REDQ-rem-walker2d': 'REDQ_embpo_qrem_piave_n10_m2_uf20_pd20_walker2d-v2',
'REDQ-ave-walker2d': 'REDQ_embpo_qave_piave_n10_m2_uf20_pd20_walker2d-v2',
'REDQ-min-walker2d': 'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_walker2d-v2',
'REDQ-rem-ant': 'REDQ_embpo_qrem_piave_n10_m2_uf20_pd20_ant-v2',
'REDQ-ave-ant': 'REDQ_embpo_qave_piave_n10_m2_uf20_pd20_ant-v2',
'REDQ-min-ant': 'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_ant-v2',
'REDQ-ave-humanoid': 'REDQ_embpo_qave_piave_n10_m2_uf20_pd20_humanoid-v2',
'REDQ-weighted-humanoid': 'REDQ_embpo_qweighted_piave_n10_m2_uf20_pd20_humanoid-v2',
'REDQ-m1-hopper':'REDQ_embpo_qmin_piave_n10_m1_uf20_pd20_hopper-v2',
'REDQ-m1-walker2d':'REDQ_embpo_qmin_piave_n10_m1_uf20_pd20_walker2d-v2',
'REDQ-m1-ant':'REDQ_embpo_qmin_piave_n10_m1_uf20_pd20_ant-v2',
'REDQ-m1-humanoid':'REDQ_embpo_qmin_piave_n10_m1_uf20_pd20_humanoid-v2',
}

exp2path_M_ablation = {
'REDQ-n10-m1-5-ant': 'REDQ_embpo_qmin_piave_n10_m1-5_uf20_pd20_ant-v2',
'REDQ-n10-m2-5-ant': 'REDQ_embpo_qmin_piave_n10_m2-5_uf20_pd20_ant-v2',
'REDQ-n10-m3-ant': 'REDQ_embpo_qmin_piave_n10_m3_uf20_pd20_ant-v2',
'REDQ-n10-m5-ant': 'REDQ_embpo_qmin_piave_n10_m5_uf20_pd20_ant-v2',
'REDQ-n10-m1-5-walker2d': 'REDQ_embpo_qmin_piave_n10_m1-5_uf20_pd20_walker2d-v2',
'REDQ-n10-m2-5-walker2d': 'REDQ_embpo_qmin_piave_n10_m2-5_uf20_pd20_walker2d-v2',
'REDQ-n10-m3-walker2d': 'REDQ_embpo_qmin_piave_n10_m3_uf20_pd20_walker2d-v2',
'REDQ-n10-m5-walker2d': 'REDQ_embpo_qmin_piave_n10_m5_uf20_pd20_walker2d-v2',
'REDQ-n10-m1-5-hopper': 'REDQ_embpo_qmin_piave_n10_m1-5_uf20_pd20_hopper-v2',
'REDQ-n10-m2-5-hopper': 'REDQ_embpo_qmin_piave_n10_m2-5_uf20_pd20_hopper-v2',
'REDQ-n10-m3-hopper': 'REDQ_embpo_qmin_piave_n10_m3_uf20_pd20_hopper-v2',
'REDQ-n10-m5-hopper': 'REDQ_embpo_qmin_piave_n10_m5_uf20_pd20_hopper-v2',
}

exp2path_ablations_app = {
'REDQ-ave-n15-ant': 'REDQ_embpo_qave_piave_n15_m2_uf20_pd20_ant-v2',
'REDQ-ave-n5-ant': 'REDQ_embpo_qave_piave_n5_m2_uf20_pd20_ant-v2',
'REDQ-ave-n3-ant': 'REDQ_embpo_qave_piave_n3_m2_uf20_pd20_ant-v2',
'REDQ-ave-n2-ant': 'REDQ_embpo_qave_piave_n2_m2_uf20_pd20_ant-v2',
'REDQ-ave-n15-walker2d': 'REDQ_embpo_qave_piave_n15_m2_uf20_pd20_walker2d-v2',
'REDQ-ave-n5-walker2d': 'REDQ_embpo_qave_piave_n5_m2_uf20_pd20_walker2d-v2',
'REDQ-ave-n3-walker2d': 'REDQ_embpo_qave_piave_n3_m2_uf20_pd20_walker2d-v2',
'REDQ-ave-n2-walker2d': 'REDQ_embpo_qave_piave_n2_m2_uf20_pd20_walker2d-v2',
}

exp2path_utd = {
'REDQ-n10-utd1-ant':'REDQ_embpo_qmin_piave_n10_m2_uf1_pd20_ant-v2',
'REDQ-n10-utd5-ant':'REDQ_embpo_qmin_piave_n10_m2_uf5_pd20_ant-v2',
'REDQ-n10-utd10-ant':'REDQ_embpo_qmin_piave_n10_m2_uf10_pd20_ant-v2',
'REDQ-n10-utd1-walker2d':'REDQ_embpo_qmin_piave_n10_m2_uf1_pd20_walker2d-v2',
'REDQ-n10-utd5-walker2d':'REDQ_embpo_qmin_piave_n10_m2_uf5_pd20_walker2d-v2',
'REDQ-n10-utd10-walker2d':'REDQ_embpo_qmin_piave_n10_m2_uf10_pd20_walker2d-v2',
'SAC-5-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf5_pd20_ant-v2', # SAC with policy delay
'SAC-5-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf5_pd20_walker2d-v2',
'SAC-10-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf10_pd20_ant-v2',
'SAC-10-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf10_pd20_walker2d-v2',
'REDQ-min-n3-utd1-ant':'REDQ_embpo_qmin_piave_n3_msame_uf1_pd20_ant-v2',
'REDQ-min-n3-utd5-ant':'REDQ_embpo_qmin_piave_n3_msame_uf5_pd20_ant-v2',
'REDQ-min-n3-utd10-ant':'REDQ_embpo_qmin_piave_n3_msame_uf10_pd20_ant-v2',
'REDQ-min-n3-utd20-ant':'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_ant-v2',
'REDQ-min-n3-utd1-walker2d':'REDQ_embpo_qmin_piave_n3_msame_uf1_pd20_walker2d-v2',
'REDQ-min-n3-utd5-walker2d':'REDQ_embpo_qmin_piave_n3_msame_uf5_pd20_walker2d-v2',
'REDQ-min-n3-utd10-walker2d':'REDQ_embpo_qmin_piave_n3_msame_uf10_pd20_walker2d-v2',
'REDQ-min-n3-utd20-walker2d':'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_walker2d-v2',
'REDQ-min-n3-utd1-hopper':'REDQ_embpo_qmin_piave_n3_msame_uf1_pd20_hopper-v2',
'REDQ-min-n3-utd5-hopper':'REDQ_embpo_qmin_piave_n3_msame_uf5_pd20_hopper-v2',
'REDQ-min-n3-utd10-hopper':'REDQ_embpo_qmin_piave_n3_msame_uf10_pd20_hopper-v2',
'REDQ-min-n3-utd20-hopper':'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_hopper-v2',
'REDQ-min-n3-utd1-humanoid':'REDQ_embpo_qmin_piave_n3_msame_uf1_pd20_humanoid-v2',
'REDQ-min-n3-utd5-humanoid':'REDQ_embpo_qmin_piave_n3_msame_uf5_pd20_humanoid-v2',
'REDQ-min-n3-utd10-humanoid':'REDQ_embpo_qmin_piave_n3_msame_uf10_pd20_humanoid-v2',
'REDQ-min-n3-utd20-humanoid':'REDQ_embpo_qmin_piave_n3_msame_uf20_pd20_humanoid-v2',
'REDQ-ave-utd1-ant':'REDQ_embpo_qave_piave_n10_m2_uf1_pd20_ant-v2',
'REDQ-ave-utd5-ant':'REDQ_embpo_qave_piave_n10_m2_uf5_pd20_ant-v2',
'REDQ-ave-utd1-walker2d':'REDQ_embpo_qave_piave_n10_m2_uf1_pd20_walker2d-v2',
'REDQ-ave-utd5-walker2d':'REDQ_embpo_qave_piave_n10_m2_uf5_pd20_walker2d-v2',
}

exp2path_pd = {
'SAC-pd1-5-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf5_pd1_ant-v2', # no policy delay SAC
'SAC-pd1-5-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf5_pd1_walker2d-v2',
'SAC-pd1-10-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf10_pd1_ant-v2',
'SAC-pd1-10-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf10_pd1_walker2d-v2',
'SAC-pd1-20-ant': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd1_ant-v2',
'SAC-pd1-20-walker2d': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd1_walker2d-v2',
'SAC-pd1-20-hopper': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd1_hopper-v2',
'SAC-pd1-20-humanoid': 'REDQ_embpo_qmin_piave_n2_m2_uf20_pd1_humanoid-v2',
'REDQ-n10-pd1-ant':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd1_ant-v2',
'REDQ-n10-pd1-walker2d':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd1_walker2d-v2',
'REDQ-n10-pd1-hopper':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd1_hopper-v2',
'REDQ-n10-pd1-humanoid':'REDQ_embpo_qmin_piave_n10_m2_uf20_pd1_humanoid-v2',
}

exp2path_ofe = {
    'REDQ-ofe-humanoid': 'REDQ_e20_uf20_pd_ofe20000_p100000_d5_lr0-0003_wd5e-05_w1_humanoid-v2',
    'REDQ-ofe-ant': 'REDQ_e20_uf20_pd_ofe20000_p100000_d5_lr0-0003_wd5e-05_w1_ant-v2',
}

def generate_exp2path(name_prefix, list_values, list_name_suffix, pre_string, env_names):
    exp2path_dict = {}
    for e in env_names:
        for i in range(len(list_values)):
            new_name = '%s-%s-%s' % (name_prefix, list_name_suffix[i], e)
            new_string = pre_string % (list_values[i], e)
            exp2path_dict[new_name] = new_string
    return exp2path_dict

figsize=(10, 7)
exp2dataset = {}
############# finalizing plots ####################################
default_smooth = 10
all_4_envs = ['ant', 'humanoid', 'hopper', 'walker2d']
just_2_envs = ['ant', 'walker2d']
y_types_all = ['Performance', 'AverageQBias', 'StdQBias', 'AllNormalizedAverageQBias',  'AllNormalizedStdQBias',
               'LossQ1', 'NormLossQ1', 'AverageQBiasSqr', 'MaxPreTanh', 'AveragePreTanh', 'AverageQ1Vals',
               'MaxAlpha', 'AverageAlpha', 'AverageQPred']
y_types_appendix_6 = ['AverageQBias', 'StdQBias', 'AverageQBiasSqr', 'AllNormalizedAverageQBiasSqr',
                      'LossQ1', 'AverageQPred']
y_types_main_paper = ['Performance', 'AllNormalizedAverageQBias', 'AllNormalizedStdQBias']
y_types_appendix_9 = y_types_main_paper + y_types_appendix_6

# main paper
plot_result_sec = True
plot_analysis_sec = True
plot_ablations = True
plot_variants = True
plot_ablations_revision = True
# appendix ones
plot_analysis_app = True
plot_ablations_app = True
plot_variants_app = True
plot_utd_app = True
plot_pd_app = True
plot_ofe = True

if plot_ablations_app:
    exp2dataset.update(get_exp2dataset(exp2path_ablations_app, base_path))
if plot_utd_app:
    exp2dataset.update(get_exp2dataset(exp2path_utd, base_path))
if plot_pd_app:
    exp2dataset.update(get_exp2dataset(exp2path_pd, base_path))
if plot_ofe:
    exp2dataset.update(get_exp2dataset(exp2path_ofe, base_path))

exp2dataset.update(get_exp2dataset(exp2path_main, base_path))
exp2dataset.update(get_exp2dataset(exp2path_N_ablation, base_path))
exp2dataset.update(get_exp2dataset(exp2path_M_ablation, base_path))
exp2dataset.update(get_exp2dataset(exp2path_redq_variants, base_path))

if plot_result_sec:
    # 1. redq-mbpo figure in results section, score only
    exp_base_to_plot = ['REDQ-n10', 'SAC-1', 'MBPO']
    save_path, prefix = 'results', 'redq-mbpo'
    label_list = ['REDQ', 'SAC', 'MBPO']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              all_4_envs, ['Performance'], default_smooth, figsize, label_list=label_list, legend_y_types=['Performance'],
              legend_es=['humanoid'])

if plot_analysis_sec:
    # 2. analysis sec, REDQ, SAC-20, ave comparison, argue naive methods don't work
    exp_base_to_plot = ['REDQ-n10', 'SAC-20', 'REDQ-ave']
    save_path, prefix = 'analysis', 'redq-sac'
    label_list = ['REDQ', 'SAC-20', 'AVG']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              all_4_envs, y_types_all, default_smooth, figsize, label_list=label_list,
              legend_y_types=['AllNormalizedStdQBias'], legend_es='all', legend_loc='upper right')

if plot_ablations:
    # 3. abaltion on N
    exp_base_to_plot = ['REDQ-n15', 'REDQ-n10', 'REDQ-n5', 'REDQ-n3', 'REDQ-n2']
    save_path, prefix = 'ablations', 'redq-N'
    label_list = ['REDQ-N15', 'REDQ-N10', 'REDQ-N5', 'REDQ-N3', 'REDQ-N2']
    overriding_ylimit_dict = {
        ('ant', 'AllNormalizedAverageQBias'): (-1, 1),
        ('ant', 'AllNormalizedStdQBias'): (0, 1),
        ('walker2d', 'AllNormalizedAverageQBias'): (-0.5, 0.75),
        ('walker2d', 'AllNormalizedStdQBias'):(0, 0.75),
    }
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_main_paper, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AllNormalizedStdQBias'], legend_es=['ant'], legend_loc='upper right',
              overriding_ylimit_dict=overriding_ylimit_dict)

    # 4. abaltion on M
    exp_base_to_plot = ['REDQ-n10-m1-5', 'REDQ-n10', 'REDQ-n10-m2-5', 'REDQ-n10-m3', 'REDQ-n10-m5']
    save_path, prefix = 'ablations', 'redq-M'
    label_list = ['REDQ-M1.5', 'REDQ-M2', 'REDQ-M2.5', 'REDQ-M3', 'REDQ-M5']
    overriding_ylimit_dict = {
        ('ant', 'AllNormalizedAverageQBias'): (-4, 4),
         ('walker2d', 'AllNormalizedAverageQBias'): (-1, 2.5),
         }
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_main_paper, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AllNormalizedStdQBias'], legend_es=['ant'], legend_loc='upper right',
              overriding_ylimit_dict=overriding_ylimit_dict)

if plot_ablations_revision:
    # 5. different variant of Q target computation comparison (here maybe we should do... )
    exp_base_to_plot = ['REDQ-n10', 'REDQ-weighted']
    save_path, prefix = 'revision', 'redq-weighted'
    label_list = ['REDQ', 'Weighted']
    overriding_ylimit_dict = {
        ('ant', 'AllNormalizedAverageQBias'): (-1, 1),
        # ('hopper', 'AllNormalizedAverageQBias'): (-0.5, 1.5),
        # ('humanoid', 'AllNormalizedAverageQBias'): (-0.5, 0),
        # ('walker2d', 'AllNormalizedAverageQBias'): (-0.5, 0.5),
    }
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              all_4_envs, y_types_main_paper, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AllNormalizedStdQBias'], legend_es='all', legend_loc='upper right',
              overriding_ylimit_dict=overriding_ylimit_dict)

if plot_variants:
    # 5. different variant of Q target computation comparison (here maybe we should do... )
    exp_base_to_plot = ['REDQ-n10', 'REDQ-rem', 'REDQ-min', 'REDQ-weighted', 'REDQ-minpair']
    save_path, prefix = 'variants', 'redq-var'
    label_list = ['REDQ', 'REM', 'Maxmin', 'Weighted', 'MinPair']
    overriding_ylimit_dict = {
        ('ant', 'AllNormalizedAverageQBias'): (-2, 4),
         }
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_main_paper, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AllNormalizedStdQBias'], legend_es=['ant'], legend_loc='upper right',
              overriding_ylimit_dict=overriding_ylimit_dict)

def plot_grid(save_path, save_name_prefix, exp2dataset, exp_base_to_plot, envs, y_types, smooth, figsize, label_list=None,
              legend_y_types='all', legend_es='all', overriding_ylimit_dict=None, legend_loc='best', longxaxis=False,
              linestyle_list=None, color_list=None, override_xlimit=None):
    for y_i, y_type in enumerate(y_types):
        y_save_name = y2savename_dict[y_type]
        for e in envs:
            no_legend = decide_no_legend(legend_y_types, legend_es, y_type, e)
            exp_to_plot = []
            for exp_base in exp_base_to_plot:
                exp_to_plot.append(exp_base + '-' + e)
            if override_xlimit is None:
                xlimit = 125000 if e == 'hopper' else int(3e5)
            else:
                xlimit = override_xlimit
            if xlimit is None and longxaxis:
                xlimit = int(2e6)
            ylimit = get_ylimit_from_env_ytype(e, y_type, overriding_ylimit_dict)
            assert len(label_list) == len(exp_base_to_plot)
            if save_path:
                save_name = save_name_prefix + '-%s-%s' %(e, y_save_name)
            else:
                save_name = None
            if color_list is None:
                color_list = get_default_colors(exp_base_to_plot)
            plot_from_data(None, exp2dataset, exp_to_plot, figsize, value_list=[y_type,],
                           ylabel='auto', save_path=save_path, label_list=label_list,
                           save_name=save_name, smooth=smooth, xlimit=xlimit, y_limit = ylimit,
                           color_list=color_list, no_legend=no_legend, legend_loc=legend_loc,
                           linestyle_list=linestyle_list)

if plot_ofe:
    exp_base_to_plot = ['REDQ-n10', 'SAC-1', 'MBPO', 'REDQ-ofe']
    save_path, prefix = 'extra', 'redq-ofe'
    label_list = ['REDQ', 'SAC', 'MBPO', 'REDQ-OFE']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              ['humanoid', 'ant'], ['Performance'], default_smooth, figsize,
              label_list=label_list,  legend_y_types='all', legend_es=['humanoid'], legend_loc='lower right')

"""####################################  REST IS APPENDIX  #################################### """
if plot_analysis_app:
    # appendix: extra figures for 2. analysis
    exp_base_to_plot = ['REDQ-n10', 'SAC-20', 'REDQ-ave']
    save_path, prefix = 'analysis_app', 'redq-sac'
    label_list = ['REDQ', 'SAC-20', 'AVG']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_all, default_smooth, figsize, label_list=label_list,
              legend_y_types=['AverageQPred'], legend_es='all', legend_loc='upper left')

if plot_ablations_app:
    # appendix: extra figures for N, M comparisons
    # app abaltion on N
    exp_base_to_plot = ['REDQ-n15', 'REDQ-n10', 'REDQ-n5', 'REDQ-n3', 'REDQ-n2']
    save_path, prefix = 'ablations_app', 'redq-N'
    label_list = ['REDQ-N15', 'REDQ-N10', 'REDQ-N5', 'REDQ-N3', 'REDQ-N2']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_appendix_6, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AverageQPred'], legend_es='all', legend_loc='upper left',)

    # app abaltion on M
    exp_base_to_plot = ['REDQ-n10-m1-5', 'REDQ-n10', 'REDQ-n10-m2-5', 'REDQ-n10-m3', 'REDQ-n10-m5']
    save_path, prefix = 'ablations_app', 'redq-M'
    label_list = ['REDQ-M1.5', 'REDQ-M2', 'REDQ-M2.5', 'REDQ-M3', 'REDQ-M5']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_appendix_6, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AverageQPred'], legend_es='all', legend_loc='upper left',)

if plot_variants_app:
    # appendix: extra figures for variants
    exp_base_to_plot = ['REDQ-n10', 'REDQ-rem', 'REDQ-min', 'REDQ-weighted', 'REDQ-minpair']
    save_path, prefix = 'variants_app', 'redq-var'
    label_list = ['REDQ', 'REM', 'MIN', 'Weighted', 'MinPair']
    overriding_ylimit_dict = {
        ('ant', 'AllNormalizedAverageQBias'): (-2, 4),
         }
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_appendix_6, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['AverageQPred'], legend_es='all', legend_loc='upper left',
              overriding_ylimit_dict=overriding_ylimit_dict)

if plot_utd_app:
    # REDQ different utd
    exp_base_to_plot = [ 'REDQ-n10', 'REDQ-n10-utd10', 'REDQ-n10-utd5', 'REDQ-n10-utd1', ]
    save_path, prefix = 'utd_app', 'utd-redq'
    label_list = [ 'REDQ-UTD20','REDQ-UTD10','REDQ-UTD5' , 'REDQ-UTD1',]
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_all, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['Performance'], legend_es='all', legend_loc='upper left',)
    # SAC different utd
    exp_base_to_plot = ['SAC-20', 'SAC-10', 'SAC-5', 'SAC-1']
    save_path, prefix = 'utd_app', 'utd-sac'
    label_list = ['SAC-UTD20', 'SAC-UTD10', 'SAC-UTD5', 'SAC-UTD1',]
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              just_2_envs, y_types_all, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['Performance'], legend_es='all', legend_loc='upper left',)

if plot_pd_app:
    exp_base_to_plot = ['REDQ-n10', 'REDQ-n10-pd1', 'SAC-20', 'SAC-pd1-20']
    save_path, prefix = 'pd_app', 'pd-redq'
    label_list = ['REDQ','REDQ-NPD','SAC-20', 'SAC-20-NPD',]
    linestyle_list = ['solid', 'dashed', 'solid', 'dashed']
    plot_grid(save_path, prefix, exp2dataset, exp_base_to_plot,
              all_4_envs, y_types_all, default_smooth, figsize,
              label_list=label_list,  legend_y_types=['Performance'], legend_es='all', legend_loc='upper left',
              linestyle_list=linestyle_list)
