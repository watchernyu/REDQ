"""
NOTE: currently only works with seaborn 0.8.1 the tsplot function is deprecated in the newer version
the plotting function is originally based on the plot function in OpenAI spinningup
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from packaging import version

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    the "condition" here can be a string, when plotting, can be used as label on the legend
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                exp_name = None
                try:
                    config_path = open(os.path.join(root, 'config.json'))
                    config = json.load(config_path)
                    if 'exp_name' in config:
                        exp_name = config['exp_name']
                except:
                    print('No file named config.json')
                condition1 = condition or exp_name or 'exp'
                condition2 = condition1 + '-' + str(exp_idx)
                exp_idx += 1
                if condition1 not in units:
                    units[condition1] = 0
                unit = units[condition1]
                units[condition1] += 1

                print(os.path.join(root, 'progress.txt'))

                # exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
                exp_data.insert(len(exp_data.columns), 'Unit', unit)
                exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
                exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
                if performance in exp_data:
                    exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
                datasets.append(exp_data)
            except Exception as e:
                print(e)

    return datasets


y2savename_dict = {
'Performance':'score',
'AverageNormQBias':'bias-ave-n',
'StdNormQBias':'bias-std-n',
'LossQ1':'qloss',
'NormLossQ1':'qloss-n',
'MaxPreTanh':'pretanh-max',
'AveragePreTanh':'pretanh-ave',
'AverageQBias':'bias-ave',
'StdQBias':'bias-std',
'AverageQPred':'qpred',
'AverageQBiasSqr':'biassqr-ave',
'AverageNormQBiasSqr':'biassqr-ave-n',
'AverageAlpha':'alpha-ave',
'MaxAlpha':'alpha',
'AverageQ1Vals':'q1',
    'AverageLogPi':'average-logpi',
    'MaxLogPi': 'max-logpi',
    'AllNormalizedAverageQBias':'bias-ave-alln',
    'AllNormalizedStdQBias':'bias-std-alln',
    'AllNormalizedAverageQBiasSqr':'biassqr-ave-alln',
    'Time':'time'
}

y2ylabel_dict = {
'Performance':'Average return',
'AverageNormQBias':'Average normalized bias',
'StdNormQBias':'Std of normalized bias',
'LossQ1':'Q loss',
'NormLossQ1':'Normalized Q loss',
'MaxPreTanh':'Max pretanh',
'AveragePreTanh':'Average pretanh',
'AverageQBias':'Average bias',
'StdQBias':'Std of bias',
'AverageQPred':'Average Q value',
'AverageQBiasSqr':'Average MSE',
'AverageNormQBiasSqr':'Average normalized MSE',
'AverageAlpha':'Average alpha',
'MaxAlpha':'Max alpha',
'AverageQ1Vals':'Q value',
    'AverageLogPi': 'Average logPi',
    'MaxLogPi': 'Max logPi',
    'AllNormalizedAverageQBias': 'Average normalized bias',
    'AllNormalizedStdQBias': 'Std of normalized bias',
    'AllNormalizedAverageQBiasSqr': 'Average normalized MSE',
     'Time':'Time'
}

# we can use strict mapping from exp base name to color
expbase2color = {
    'SAC-20': 'grey',
    'SAC-10': 'slateblue',
    'SAC-5': 'blue',
    'SAC-1': 'skyblue', # blue-black for SAC, MBPO
    'SAC-hs1':'black',
    'SAC-hs2':'brown',
    'SAC-hs3':'purple',
    'SAC-pd1-5': 'black',
    'SAC-pd1-10': 'brown',
    'SAC-pd1-20': 'grey',
    'MBPO':'tab:blue',
    'REDQ-n15':'tab:orange', # dark red to light purple
    'REDQ-n10':'tab:red',
    'REDQ-n5': 'tab:cyan',
    'REDQ-n3': 'tab:grey',
    'REDQ-n2': 'black',
    'REDQ-n10-m1-5':'tab:orange', # red-brown for M variants?
    'REDQ-n10-m2-5':'tab:cyan',
    'REDQ-n10-m3':'tab:grey',
    'REDQ-n10-m5': 'black',
    'REDQ-n10-hs1': 'tab:orange',
    'REDQ-n10-hs2': 'violet',
    'REDQ-n10-hs3': 'lightblue',
    'REDQ-weighted': 'indigo', # then for redq q target variants, some random stuff
    'REDQ-minpair': 'royalblue',
    'REDQ-ave': 'peru',
    'REDQ-rem': 'yellow',
    'REDQ-min': 'slategrey',
    'REDQ-n10-utd10': 'tab:cyan',
    'REDQ-n10-utd5': 'tab:grey',
    'REDQ-n10-utd1':  'black',
    'REDQ-min-n3-utd1':'blue',
    'REDQ-min-n3-utd5':'lightblue',
    'REDQ-min-n3-utd10':'grey',
    'REDQ-min-n3-utd20':'black',
    'REDQ-ave-utd1':'blue',
    'REDQ-ave-utd5':'grey',
    'REDQ-ofe':'purple',
    'REDQ-ofe-long':'purple',
    'REDQ-dense': 'deeppink',
    'SAC-dense': 'sandybrown',
    'REDQ-ave-n15':'black',
'REDQ-ave-n5':'tab:orange',
'REDQ-ave-n3':'violet',
'REDQ-ave-n2':'lightblue',
    'SAC-long':'skyblue',
    'REDQ-n10-pd1':'tab:red',
    'REDQ-fine':'tab:pink',
    'REDQ-ss10k':'tab:pink',
    'REDQ-ss15k': 'tab:orange',
    'REDQ-ss20k': 'yellow',
    'REDQ-weighted-ss10k':'lightblue',
    'REDQ-weighted-ss15k': 'royalblue',
    'REDQ-weighted-ss20k': 'black',
    'REDQ-m1':'grey',
    'REDQ-more':'tab:red',
    'REDQ-weighted-more':'indigo',
    'REDQ-anneal25k': 'tab:pink',
    'REDQ-anneal50k': 'tab:orange',
    'REDQ-anneal100k': 'yellow',
    'REDQ-weighted-anneal25k': 'lightblue',
    'REDQ-weighted-anneal50k': 'royalblue',
    'REDQ-weighted-anneal100k': 'black',
    'REDQ-init0-4': 'tab:pink',
    'REDQ-init0-3': 'tab:orange',
    'REDQ-init0-2': 'yellow',
    'REDQ-weighted-init0-4': 'lightblue',
    'REDQ-weighted-init0-3': 'royalblue',
    'REDQ-weighted-init0-2': 'black',
    'REDQ-weighted-a0-1':'lightblue',
    'REDQ-weighted-a0-15': 'royalblue',
     'REDQ-weighted-a0-2':'black',
    'REDQ-weighted-a0-25':'pink',
     'REDQ-weighted-a0-3':'yellow',
    'REDQ-a0-1': 'lightblue',
    'REDQ-a0-15': 'royalblue',
    'REDQ-a0-2': 'black',
    'REDQ-a0-25': 'pink',
    'REDQ-a0-3': 'yellow',
    'REDQ-weighted-wns0-1': 'lightblue',
    'REDQ-weighted-wns0-3': 'royalblue',
    'REDQ-weighted-wns0-5': 'black',
    'REDQ-weighted-wns0-8': 'pink',
    'REDQ-weighted-wns1-2': 'yellow',
}

def get_ylimit_from_env_ytype(e, ytype, overriding_dict=None):
    # will map env and ytype to ylimit
    # so that the plot looks better
    # the plot should give a reasonable y range so that the outlier won't dominate
    # can use a overriding dictionary to override the default values here
    default_dict = {
        ('ant', 'AverageQBias'): (-200, 300),
        ('ant', 'AverageNormQBias'): (-1, 4),
        ('ant', 'AllNormalizedAverageQBias'): (-1, 4),
        ('ant', 'StdQBias'): (0, 150),
        ('ant', 'StdNormQBias'): (0, 5),
        ('ant', 'AllNormalizedStdQBias'): (0, 2),
        ('ant', 'LossQ1'): (0, 60),
         ('ant', 'NormLossQ1'): (0, 0.4),
         ('ant', 'AverageQBiasSqr'): (0, 50000),
         ('ant', 'AllNormalizedAverageQBiasSqr'): (0, 400),
        ('ant', 'AverageNormQBiasSqr'): (0, 1200),
        ('ant', 'AverageQPred'): (-200, 1000),
        ('walker2d', 'AverageQBias'): (-150, 250),
         ('walker2d', 'AverageNormQBias'): (-0.5, 4),
        ('walker2d', 'AllNormalizedAverageQBias'): (-0.5, 2.5),
        ('walker2d', 'StdQBias'): (0, 150),
         ('walker2d', 'StdNormQBias'): (0, 4),
        ('walker2d', 'AllNormalizedStdQBias'):(0, 1.5),
        ('walker2d', 'LossQ1'): (0, 60),
         ('walker2d', 'NormLossQ1'): (0, 0.5),
         ('walker2d', 'AverageQBiasSqr'): (0, 50000),
         ('walker2d', 'AverageNormQBiasSqr'): (0, 1200),
        ('walker2d', 'AllNormalizedAverageQBiasSqr'): (0, 1200),
        ('walker2d', 'AverageQPred'): (-100, 600),
        ('humanoid', 'AllNormalizedAverageQBias'): (-1, 2.5),
        ('humanoid', 'AllNormalizedStdQBias'): (0, 1.5),
    }
    key = (e, ytype)
    if overriding_dict and key in overriding_dict:
        return overriding_dict[key]
    else:
        if key in default_dict:
            return default_dict[key]
        else:
            return None

def select_data_list_for_plot(exp2dataset, exp_to_plot):
    data_list = []
    for exp in exp_to_plot:
        # use append here, since we expect exp2dataset[exp] to be a single pandas df
        data_list.append(exp2dataset[exp])
    return data_list # now a list of list

def plot_from_data(data_list, exp2dataset=None, exp_to_plot=None, figsize=None, xaxis='TotalEnvInteracts',
                   value_list=['Performance',], color_list=None, linestyle_list=None, label_list=None,
                   count=False,
                   font_scale=1.5, smooth=1, estimator='mean', no_legend=False,
                   legend_loc='best', title=None, save_name=None, save_path = None,
                   xlimit=-1, y_limit=None, label_font_size=24, xlabel=None, ylabel=None,
                   y_log_scale=False):
    """
    either give a data list
    or give a exp2dataset dictionary, and then specify the exp_to_plot
    will plot each experiment setting in data_list to the same figure.
    the label will be basically 'Condition1' column by default
    causal_plot will not care about order and will be messy and each time the order can be different.
    for value list if it contains only one value then we will use that for all experiments
    """
    if data_list is not None:
        data_list = data_list
    else:
        data_list = select_data_list_for_plot(exp2dataset, exp_to_plot)

    n_curves = len(data_list)
    if len(value_list) == 1:
        value_list = [value_list[0] for _ in range(n_curves)]
    default_colors = ['tab:blue','tab:orange','tab:green','tab:red',
                      'tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan',]
    color_list = default_colors if color_list is None else color_list
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    plt.figure(figsize=figsize) if figsize else plt.figure()
    ##########################
    value_list_smooth_temp = []
    """
    smooth data with moving window average.
    that is,
        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    where the "smooth" param is width of that window (2k+1)
    IF WE MODIFY DATA DIRECTLY, THEN CAN LEAD TO PLOTTING BUG WHERE
    IT'S MODIFIED MULTIPLE TIMES
    """
    y = np.ones(smooth)
    for i, data_seeds in enumerate(data_list):
        temp_value_name = value_list[i] + '__smooth_temp'
        value_list_smooth_temp.append(temp_value_name)
        for data in data_seeds:
            x = np.asarray(data[value_list[i]].copy())
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            # data[value_list[i]] = smoothed_x # this can be problematic
            if temp_value_name not in data:
                data.insert(len(data.columns), temp_value_name, smoothed_x)
            else:
                data[temp_value_name] = smoothed_x

    sns.set(style="darkgrid", font_scale=font_scale)
    # sns.set_palette('bright')
    # have the same axis (figure), plot one by one onto it
    ax = None
    if version.parse(sns.__version__) <= version.parse('0.8.1'):
        for i, data_seeds in enumerate(data_list):
            data_combined = pd.concat(data_seeds, ignore_index=True)
            ax = sns.tsplot(data=data_combined, time=xaxis, value=value_list_smooth_temp[i], unit="Unit",
                            condition=condition,
                            legend=(not no_legend), ci='sd',
                            n_boot=0, color=color_list[i], ax=ax)
    else:
        print("Error: Seaborn version > 0.8.1 is currently not supported.")
        quit()

    if linestyle_list is not None:
        for i in range(len(linestyle_list)):
            ax.lines[i].set_linestyle(linestyle_list[i])
    if label_list is not None:
        for i in range(len(label_list)):
            ax.lines[i].set_label(label_list[i])
    xlabel = 'environment interactions' if xlabel is None else xlabel

    if ylabel is None:
        ylabel = 'average test return'
    elif ylabel == 'auto':
        if value_list[0] in y2ylabel_dict:
            ylabel = y2ylabel_dict[value_list[0]]
        else:
            ylabel = value_list[0]
    else:
        ylabel = ylabel
    plt.xlabel(xlabel, fontsize=label_font_size)
    plt.ylabel(ylabel, fontsize=label_font_size)
    if not no_legend:
        plt.legend(loc=legend_loc, fontsize=label_font_size)

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if y_log_scale:
        plt.yscale('log')
    if xlimit > 0:
        plt.xlim(0, xlimit)
    if y_limit:
        plt.ylim(y_limit[0], y_limit[1])
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_name is not None:
        fig = plt.gcf()
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        fig.savefig(os.path.join(save_path, save_name))
        plt.close(fig)
    else:
        plt.show()


def decide_no_legend(legend_y_types, legend_es, y_type, e):
    # return no_legend as a bool (=True means no legend will be plotted)
    if (legend_y_types == 'all' or y_type in legend_y_types) and (legend_es == 'all' or e in legend_es):
        return False
    else:
        return True

def plot_grid(save_path, save_name_prefix, exp2dataset, exp_base_to_plot, envs, y_types, smooth, figsize, label_list=None,
              legend_y_types='all', legend_es='all', overriding_ylimit_dict=None, legend_loc='best', longxaxis=False,
              linestyle_list=None, color_list=None):
    """
    this function will help us do generic redq plots very easily and fast.
    Args:
        save_path: where to save the figures (i.e. 'figures')
        envs: specify the envs that will be plotted (i.e. 'ant')
        exp_base_to_plot: base name of the exp settings (i.e. 'REDQ-n10')
        y_types: y type to be plotting (for example, 'Performance')
        smooth:
    """
    for y_i, y_type in enumerate(y_types):
        y_save_name = y2savename_dict[y_type]
        for e in envs:
            no_legend = decide_no_legend(legend_y_types, legend_es, y_type, e)
            exp_to_plot = []
            for exp_base in exp_base_to_plot:
                exp_to_plot.append(exp_base + '-' + e)
            xlimit = 125000 if e == 'hopper' else int(3e5)
            if longxaxis:
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

def plot_grid_solid_dashed(save_path, save_name_prefix, exp2dataset, exp_base_to_plot, envs, y_1, y_2, smooth, figsize, label_list=None):
    """
    this function will help us do generic redq plots very easily and fast.
    Args:
        save_path: where to save the figures (i.e. 'figures')
        envs: specify the envs that will be plotted (i.e. 'ant')
        exp_base_to_plot: base name of the exp settings (i.e. 'REDQ-n10')
        y_types: y type to be plotting (for example, 'Performance')
        smooth:
    """
    y_save_name = y2savename_dict[y_1] # typically this is Q value, then y2 is bias
    for e in envs:
        exp_to_plot = []
        for exp_base in exp_base_to_plot:
            exp_to_plot.append(exp_base + '-' + e)
        value_list = []
        linestyle_list = []
        for i in range(len(exp_to_plot)):
            value_list.append(y_1)
            linestyle_list.append('solid')
        for i in range(len(exp_to_plot)):
            value_list.append(y_2)
            linestyle_list.append('dashed')
        exp_to_plot = exp_to_plot + exp_to_plot
        exp_base_to_plot_for_color = exp_base_to_plot + exp_base_to_plot
        xlimit = 125000 if e == 'hopper' else int(3e5)
        ylimit = get_ylimit_from_env_ytype(e, y_1)
        plot_from_data(None, exp2dataset, exp_to_plot, figsize, value_list=value_list, linestyle_list=linestyle_list,
                       ylabel='auto', save_path=save_path, label_list=label_list,
                       save_name=save_name_prefix + '-%s-%s' %(e, y_save_name), smooth=smooth, xlimit=xlimit, y_limit = ylimit,
                       color_list=get_default_colors(exp_base_to_plot_for_color))

def get_default_colors(exp_base_names):
    # will map experiment base name (the good names) to color
    # later ones
    colors = []
    for i in range(len(exp_base_names)):
        exp_base_name = exp_base_names[i]
        colors.append(expbase2color[exp_base_name])
    return colors

def get_exp2dataset(exp2path, base_path):
    """
    Args:
        exp2path: a dictionary containing experiment name (you can decide what this is, can make it easy to read,
        will be set as "condition") to the actual experiment folder name
        base_path: the base path lead to where the experiment folders are
    Returns:
        a dictionary with keys being the experiment name, value being a pandas dataframe containing
        the experiment progress (multiple seeds in one dataframe)
    """
    exp2dataset = {}
    for key in exp2path.keys():
        complete_path = os.path.join(base_path, exp2path[key])
        data_list = get_datasets(complete_path, key) # a list of different seeds
        # combined_df_across_seeds = pd.concat(data_list, ignore_index=True)
        # exp2dataset[key] = combined_df_across_seeds
        for d in data_list:
            try:
                add_normalized_values(d)
            except:
                print("can't add normalized loss in data:", key)
        exp2dataset[key] = data_list
    return exp2dataset

def add_normalized_values(d):
    normalize_base = d['AverageMCDisRetEnt'].copy().abs()
    normalize_base[normalize_base < 10] = 10
    # use this to normalize the Q loss, Q bias

    normalized_qloss = d['LossQ1'] / normalize_base
    d.insert(len(d.columns), 'NormLossQ1', normalized_qloss)
    normalized_q_bias = d['AverageQBias'] / normalize_base
    d.insert(len(d.columns), 'AllNormalizedAverageQBias', normalized_q_bias)
    normalized_std_q_bias = d['StdQBias'] / normalize_base
    d.insert(len(d.columns), 'AllNormalizedStdQBias', normalized_std_q_bias)
    normalized_bias_mse = d['AverageQBiasSqr'] / normalize_base
    d.insert(len(d.columns), 'AllNormalizedAverageQBiasSqr', normalized_bias_mse)
