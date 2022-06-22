from train_redq_sac import redq_sac as function_to_run ## here make sure you import correct function
import time
from redq.utils.run_utils import setup_logger_kwargs
from grid_utils import get_setting_and_exp_name

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    data_dir = '../data'

    exp_prefix = 'trial'
    settings = ['env_name','',['Hopper-v4', 'Ant-v4'],
                'seed','',[0, 1, 2],
                'epochs','e',[20],
                'num_Q','q',[2],
                'utd_ratio','uf',[1],
                'policy_update_delay','pd',[20],
                ]

    indexes, actual_setting, total, exp_name_full = get_setting_and_exp_name(settings, args.setting, exp_prefix)
    print("##### TOTAL NUMBER OF VARIANTS: %d #####" % total)

    logger_kwargs = setup_logger_kwargs(exp_name_full, actual_setting['seed'], data_dir)
    function_to_run(logger_kwargs=logger_kwargs, **actual_setting)
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))
