from train_redq_sac import redq_sac as function_to_run ## here make sure you import correct function
import time
import numpy as np
from redq.utils.run_utils import setup_logger_kwargs

def get_setting_and_exp_name(settings, setting_number, exp_prefix, random_setting_seed=0, random_order=True):
    np.random.seed(random_setting_seed)
    hypers, lognames, values_list = [], [], []
    hyper2logname = {}
    n_settings = int(len(settings)/3)
    for i in range(n_settings):
        hypers.append(settings[i*3])
        lognames.append(settings[i*3+1])
        values_list.append(settings[i*3+2])
        hyper2logname[hypers[-1]] = lognames[-1]

    total = 1
    for values in values_list:
        total *= len(values)
    max_job = total

    new_indexes = np.random.choice(total, total, replace=False) if random_order else np.arange(total)
    new_index = new_indexes[setting_number]

    indexes = []  ## this says which hyperparameter we use
    remainder = new_index
    for values in values_list:
        division = int(total / len(values))
        index = int(remainder / division)
        remainder = remainder % division
        indexes.append(index)
        total = division
    actual_setting = {}
    for j in range(len(indexes)):
        actual_setting[hypers[j]] = values_list[j][indexes[j]]

    exp_name_full = exp_prefix
    for hyper, value in actual_setting.items():
        if hyper not in ['env_name', 'seed']:
            exp_name_full = exp_name_full + '_%s' % (hyper2logname[hyper] + str(value))
    exp_name_full = exp_name_full + '_%s' % actual_setting['env_name']

    return indexes, actual_setting, max_job, exp_name_full

