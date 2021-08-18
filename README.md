# REDQ source code
Author's PyTorch implementation of Randomized Ensembled Double Q-Learning (REDQ) algorithm. Paper link: https://arxiv.org/abs/2101.05982

Aug 18, 2021: **VERY IMPORTANT BUG FIX** in `experiments/train_redq_sac.py`, the done signal is not being correctly used, the done signal value should be `False` when the episode terminates due to environment timelimit, but in the earlier version of the code, 
the agent puts the transition in buffer before this value is corrected. This can affect performance especially for environments where termination due to bad action is rare. This is now fixed and we might do some more testing. If you use this file to run experiments **please check immediately or pull the latest version** of the code. 
Sorry for the bug! Please don't hesitate to open an issue if you have any questions.

July, 2021: data and the function to reproduce all figures in the paper are now available, see the `Data and reproducing figures in REDQ` section for details.

Mar 23, 2021: We have reorganized the code to make it cleaner and more readable and the first version is now released! 

Mar 29, 2021: We tested the installation process and run the code, and everything seems to be working correctly. We are now working on the implementation video tutorial, which will be released soon. 

May 3, 2021: We uploaded a video tutorial (shared via google drive), please see link below. Hope it helps! 

Code for REDQ-OFE is still being cleaned up and will be released soon (essentially the same code but with additional input from a OFENet). 

## Code structure explained
The code structure is pretty simple and should be easy to follow. 

In `experiments/train_redq_sac.py` you will find the main training loop. Here we set up the environment, initialize an instance of the `REDQSACAgent` class, specifying all the hyperparameters and train the agent. You can run this file to train a REDQ agent. 

In `redq/algos/redq_sac.py` we provide code for the `REDQSACAgent` class. If you are trying to take a look at how the core components of REDQ are implemented, the most important function is the `train()` function. 

In `redq/algos/core.py` we provide code for some basic classes (Q network, policy network, replay buffer) and some helper functions. These classes and functions are used by the REDQ agent class. 

In `redq/utils` there are some utility classes (such as a logger) and helper functions that largely have nothing to do with REDQ's core components. In `redq/utils/bias_utils.py` you can find utility functions to get bias estimation (bias estimate is computed roughly as: Monte Carlo return - current Q estimate). In `experiments/train_redq_sac.py` you can decide whether you want bias evaluation when running the experiment by setting the `evaluate_bias` flag (this will lead to some minor computation overhead). 

In `plot_utils` there are some utility functions to reproduce the figures we presented in the paper. (See the section on "Data and reproducing figures in REDQ")

## Implementation video tutorial
Here is the link to a video tutorial we created that explains the REDQ implementation in detail: 

[REDQ code explained video tutorial (Google Drive Link)](https://drive.google.com/file/d/1ZUuDK6KUqAGJFaqsM5ITZ_ZyRvThaRn_/view?usp=sharing)

## Data and reproducing figures in REDQ
The data used in the REDQ paper can be downloaded here: 
[REDQ DATA download link](https://drive.google.com/file/d/1mpdb2OXxEembO83hiAXfN8DYeSy6Gl1o/view?usp=sharing) (Google Drive Link, ~80 MB)

To reproduce the figures, first download the data, and then extract the zip file to `REDQ/data`. So now a folder called `REDQ_ICLR21` should be at this path:  `REDQ/data/REDQ_ICLR21`. 

Then you can go into the `plot_utils` folder, and run the `plot_REDQ.py` program there. You will need `seaborn==0.8.1` to run it correctly. We might update the code later so that it works for newer versions but currently seaborn newer than 0.8.1 is not supported. If you don't want to mess up existing conda or python virtual environments, you can create a new environment and simply install seaborn 0.8.1 there and use it to run the program. 

If you encounter any problem or cannot access the data (can't use google or can't download), please open an issue to let us know! Thanks! 

## Environment setup
Note: you don't need to exactly follow the tutorial here if you know well about how to install python packages. 

First create a conda environment and activate it:
```
conda create -n redq python=3.6
conda activate redq 
```

Install PyTorch (or you can follow the tutorial on PyTorch official website).
On Ubuntu (might also work on Windows but is not fully tested):
```
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
```
On OSX:
```
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch
```

Install gym (0.17.2):
```
git clone https://github.com/openai/gym.git
cd gym
git checkout b2727d6
pip install -e .
cd ..
```

Install mujoco_py (2.0.2.1): 
```
git clone https://github.com/openai/mujoco-py
cd mujoco-py
git checkout 379bb19
pip install -e . --no-cache
cd ..
```

For gym and mujoco_py, depending on your system, you might need to install some other packages, if you run into such problems, please refer to their official sites for guidance. 
If you want to test on Mujoco environments, you will also need to get Mujoco files and license from Mujoco website. Please refer to the Mujoco website for how to do this correctly. 

Clone and install this repository (Although even if you don't install it you might still be able to use the code): 
```
git clone https://github.com/watchernyu/REDQ.git
cd REDQ
pip install -e .
```

## Train an REDQ agent
To train an REDQ agent, run:
```
python experiments/train_redq_sac.py
```
On a 2080Ti GPU, running Hopper to 125K will approximately take 10-12 hours. Running Humanoid to 300K will approximately take 26 hours. 

## Implement REDQ
If you intend to implement REDQ on your codebase, please refer to the paper and the tutorial (to be released) for guidance. In particular, in Appendix B of the paper, we discussed hyperparameters and some additional implementation details. One important detail is in the beginning of the training, for the first 5000 data points, we sample random action from the action space and do not perform any updates. If you perform a large number of updates with a very small amount of data, it can lead to severe bias accumulation and can negatively affect the performance. 

For REDQ-OFE, as mentioned in the paper, for some reason adding PyTorch batch norm to OFENet will lead to divergence. So in the end we did not use batch norm in our code. 

## Reproduce the results
If you use a different PyTorch version, it might still work, however, it might be better if your version is close to the ones we used. We have found that for example, on Ant environment, PyTorch 1.3 and 1.2 give quite different results. The reason is not entirely clear. 

Other factors such as versions of other packages (for example numpy) or environment (mujoco/gym) or even types of hardware (cpu/gpu) can also affect the final results. Thus reproducing exactly the same results can be difficult. However, if the package versions are the same, when averaged over a large number of random seeds, the overall performance should be similar to those reported in the paper. 

As of Mar. 29, 2021, we have used the installation guide on this page to re-setup a conda environment and run the code hosted on this repo and the reproduced results are similar to what we have in the paper (though not exactly the same, in some environments, performance are a bit stronger and others a bit weaker). 

Please open an issue if you find any problems in the code, thanks! 

## Acknowledgement

Our code for REDQ-SAC is partly based on the SAC implementation in OpenAI Spinup (https://github.com/openai/spinningup). The current code structure is inspired by the super clean TD3 source code by Scott Fujimoto (https://github.com/sfujim/TD3). 
