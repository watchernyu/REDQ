# REDQ source code
Author's PyTorch implementation of Randomized Ensembled Double Q-Learning (REDQ) algorithm. Paper link: https://arxiv.org/abs/2101.05982

<a name="table-of-contents"/> 

## Table of Contents  
- [Table of contents](#table-of-contents)
- [Code structure explained](#code-structure)  
- [Implementation video tutorial](#video-tutorial)  
- [Data and reproducing figures in REDQ](#reproduce-figures)  
- [Train an REDQ agent](#train-redq)  
- [Implement REDQ](#implement-redq)  
- [Reproduce the results](#reproduce-results)  
- [Docker + Singularity setup, mujoco2.2.2](#setup-dockersing) 
- [Environment setup MuJoCo 2.1, v4 tasks, NYU HPC 18.04](#setup-nyuhpc-new) 
- [Environment setup MuJoCo 2.1, Ubuntu 18.04](#setup-ubuntu)  
- [Environment setup MuJoCo 2.1, NYU Shanghai HPC](#setup-nyuhpc)  
- [Environment setup (old guide, before MuJoCo 2.1)](#setup-old)  
- [Acknowledgement](#acknowledgement)

June 23, 2022: added guide for setting up with OpenAI MuJoCo v4 tasks on Slurm HPCs (not fully tested yet). Currently it seems this newer version of MuJoCo is much easier to set up compared to previous ones. 

Nov 14, 2021: **MuJoCo** is now free (thanks DeepMind!) and we now have a guide on setting up with MuJoCo 2.1 + OpenAI Gym + REDQ on a linux machine (see end of this page for newest setup guide). 

Aug 18, 2021: **VERY IMPORTANT BUG FIX** in `experiments/train_redq_sac.py`, the done signal is not being correctly used, the done signal value should be `False` when the episode terminates due to environment timelimit, but in the earlier version of the code, 
the agent puts the transition in buffer before this value is corrected. This can affect performance especially for environments where termination due to bad action is rare. This is now fixed and we might do some more testing. If you use this file to run experiments **please check immediately or pull the latest version** of the code. 
Sorry for the bug! Please don't hesitate to open an issue if you have any questions.

July, 2021: data and the function to reproduce all figures in the paper are now available, see the `Data and reproducing figures in REDQ` section for details.

Mar 23, 2021: We have reorganized the code to make it cleaner and more readable and the first version is now released! 

Mar 29, 2021: We tested the installation process and run the code, and everything seems to be working correctly. We are now working on the implementation video tutorial, which will be released soon. 

May 3, 2021: We uploaded a video tutorial (shared via google drive), please see link below. Hope it helps! 

Code for REDQ-OFE is still being cleaned up and will be released soon (essentially the same code but with additional input from a OFENet). 

<a name="code-structure"/> 

## Code structure explained
The code structure is pretty simple and should be easy to follow. 

In `experiments/train_redq_sac.py` you will find the main training loop. Here we set up the environment, initialize an instance of the `REDQSACAgent` class, specifying all the hyperparameters and train the agent. You can run this file to train a REDQ agent. 

In `redq/algos/redq_sac.py` we provide code for the `REDQSACAgent` class. If you are trying to take a look at how the core components of REDQ are implemented, the most important function is the `train()` function. 

In `redq/algos/core.py` we provide code for some basic classes (Q network, policy network, replay buffer) and some helper functions. These classes and functions are used by the REDQ agent class. 

In `redq/utils` there are some utility classes (such as a logger) and helper functions that largely have nothing to do with REDQ's core components. In `redq/utils/bias_utils.py` you can find utility functions to get bias estimation (bias estimate is computed roughly as: Monte Carlo return - current Q estimate). In `experiments/train_redq_sac.py` you can decide whether you want bias evaluation when running the experiment by setting the `evaluate_bias` flag (this will lead to some minor computation overhead). 

In `plot_utils` there are some utility functions to reproduce the figures we presented in the paper. (See the section on "Data and reproducing figures in REDQ")

<a name="video-tutorial"/> 

## Implementation video tutorial
Here is the link to a video tutorial we created that explains the REDQ implementation in detail: 

[REDQ code explained video tutorial (Google Drive Link)](https://drive.google.com/file/d/1ZUuDK6KUqAGJFaqsM5ITZ_ZyRvThaRn_/view?usp=sharing)

<a name="reproduce-figures"/> 

## Data and reproducing figures in REDQ
The data used in the REDQ paper can be downloaded here: 
[REDQ DATA download link](https://drive.google.com/file/d/1mpdb2OXxEembO83hiAXfN8DYeSy6Gl1o/view?usp=sharing) (Google Drive Link, ~80 MB)

To reproduce the figures, first download the data, and then extract the zip file to `REDQ/data`. So now a folder called `REDQ_ICLR21` should be at this path:  `REDQ/data/REDQ_ICLR21`. 

Then you can go into the `plot_utils` folder, and run the `plot_REDQ.py` program there. You will need `seaborn==0.8.1` to run it correctly. We might update the code later so that it works for newer versions but currently seaborn newer than 0.8.1 is not supported. If you don't want to mess up existing conda or python virtual environments, you can create a new environment and simply install seaborn 0.8.1 there and use it to run the program. 

If you encounter any problem or cannot access the data (can't use google or can't download), please open an issue to let us know! Thanks! 

<a name="setup-old"/> 

## Environment setup (old guide, for the newest guide, see end of this page) 

**VERY IMPORTANT**: because MuJoCo is now free, the setup guide here is slightly outdated (this is the setup we used when we run our experiments for the REDQ paper), we now provide a newer updated setup guide that uses the newest MuJoCo, please see the end of the this page. 

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

<a name="train-redq"/> 

## Train an REDQ agent
To train an REDQ agent, run:
```
python experiments/train_redq_sac.py
```
On a 2080Ti GPU, running Hopper to 125K will approximately take 10-12 hours. Running Humanoid to 300K will approximately take 26 hours. 

<a name="implement-redq"/> 

## Implement REDQ
If you intend to implement REDQ on your codebase, please refer to the paper and the [video tutorial](#video-tutorial) for guidance. In particular, in Appendix B of the paper, we discussed hyperparameters and some additional implementation details. One important detail is in the beginning of the training, for the first 5000 data points, we sample random action from the action space and do not perform any updates. If you perform a large number of updates with a very small amount of data, it can lead to severe bias accumulation and can negatively affect the performance. 

For REDQ-OFE, as mentioned in the paper, for some reason adding PyTorch batch norm to OFENet will lead to divergence. So in the end we did not use batch norm in our code. 

<a name="reproduce-results"/> 

## Reproduce the results
If you use a different PyTorch version, it might still work, however, it might be better if your version is close to the ones we used. We have found that for example, on Ant environment, PyTorch 1.3 and 1.2 give quite different results. The reason is not entirely clear. 

Other factors such as versions of other packages (for example numpy) or environment (mujoco/gym) or even types of hardware (cpu/gpu) can also affect the final results. Thus reproducing exactly the same results can be difficult. However, if the package versions are the same, when averaged over a large number of random seeds, the overall performance should be similar to those reported in the paper. 

As of Mar. 29, 2021, we have used the installation guide on this page to re-setup a conda environment and run the code hosted on this repo and the reproduced results are similar to what we have in the paper (though not exactly the same, in some environments, performance are a bit stronger and others a bit weaker). 

Please open an issue if you find any problems in the code, thanks! 

<a name="setup-dockersing"/> 

## Environment setup with MuJoCo 2.2.2 and OpenAI Gym V4 tasks, with Docker or Singularity
This is a new 2023 Guide that is based on Docker and Singularity. (currently under more testing)

Local setup: simply build a docker container with the dockerfile, or modify it to your needs, or pull it from my dockerhub: `docker pull cwatcherw/mujoco:0.7`

Then you can just mount your code repository and do a docker run. 

HPC setup: 

First time setup:
```
mkdir /scratch/$USER/.sing_cache
export SINGULARITY_CACHEDIR=/scratch/$USER/.sing_cache
echo "export SINGULARITY_CACHEDIR=/scratch/$USER/.sing_cache" >> ~/.bashrc
mkdir /scratch/$USER/sing
cd /scratch/$USER/sing 
git clone https://github.com/watchernyu/REDQ.git
```

Build singularity and run singularity:
```
module load singularity
cd /scratch/$USER/sing/
singularity build --sandbox mujoco-sandbox docker://cwatcherw/mujoco:0.7
singularity exec -B /scratch/$USER/sing/REDQ:/workspace/REDQ -B /scratch/$USER/sing/mujoco-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ /scratch/$USER/sing/mujoco-sandbox bash

singularity exec -B REDQ/:/workspace/REDQ/ mujoco-sandbox bash
```

Don't forget to use following command to tell your mujoco to use egl headless rendering if you need to do rendering or use visual input. (this is required when you run on the hpc or other headless machines.)
```
export MUJOCO_GL=egl
```

Sample command to open an interactive session for debugging: 
```
srun -p parallel --pty --mem 12000 -t 0-05:00 bash
```


<a name="setup-nyuhpc-new"/> 

## Environment setup with newest MuJoCo and OpenAI Gym V4 tasks, on the NYU Shanghai hpc cluster (system is CentOS Linux release 7.4.1708, hpc management is Slurm)
This one is the newest guide (2022 Summer) that helps you set up for Gym V4 MuJoCo tasks. And this newer version of Gym MuJoCo tasks is much easier to set up compared to previous versions. If you have limited CS background, when following these steps, make sure you don't perform extra commands in between steps. 

1. to avoid storage space issues, we will work under the scratch partition (change the `netid` to your netid), we first clone the REDQ repo and `cd` into the REDQ folder. 
```
cd /scratch/NETID/
git clone https://github.com/watchernyu/REDQ.git
cd REDQ
```

2. download MuJoCo files by running the provided script.
```
bash mujoco_download.sh
```

3. load the anaconda module (since we are on the HPC, we need to use `module load` to get certain software, instead of installing them ourselves), create and activate a conda environment using the yaml file provided. (Again don't forget to change the `netid` part, and the second line is to avoid overly long text on the terminal)
```
module load anaconda3
conda config --set env_prompt '({name})'
conda env create -f conda_env.yml --prefix /scratch/NETID/redq_env
conda activate /scratch/NETID/redq_env
```

4. install redq
```
pip install -e .
```

5. run a test script, but make sure you filled in your actual netid in `experiments/sample_hpc_script.sh` so that the script can correctly locate your conda environment. 
```
cd experiments
sbatch sample_hpc_script.sh
```

<a name="setup-ubuntu"/> 

## Environment setup with newest MuJoCo 2.1, on a Ubuntu 18.04 local machine
First download MuJoCo files, on a linux machine, we put them under ~/.mujoco:
```
cd ~
mkdir ~/.mujoco
cd ~/.mujoco
curl -O https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
```

Now we create a conda environment (you will need anaconda), and install pytorch (if you just want mujoco+gym and don't want pytorch, then skip this step)
```
conda create -y -n redq python=3.8
conda activate redq
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Now install mujoco_py: 
```
cd ~
mkdir rl_course
cd rl_course
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py/
pip install -e . --no-cache
```
A list of packages that need to be installed for linux is here https://github.com/openai/mujoco-py/blob/master/Dockerfile

Now test by running python and then `import mujoco_py`, typically you will run into some error message, check that Dockerfile to see if you are missing any of the required packages (either python package or system package). 

If mujoco works, then install REDQ: 

```
cd ~
cd rl_course
git clone https://github.com/watchernyu/REDQ.git
cd REDQ/
pip install -e .
```

Now test REDQ by running:
```
python experiments/train_redq_sac.py --debug
```
If you see training logs, then the environment should be setup correctly! 


<a name="setup-nyuhpc"/> 

## Environment setup with newest MuJoCo 2.1, on the NYU Shanghai hpc cluster (system is Linux, hpc management is Slurm)
This guide helps you set up MuJoCo and then OpenAI Gym, and then REDQ. (You can also follow the guide if you just want OpenAI Gym + MuJoCo and not REDQ, REDQ is only the last step). This likely also works for NYU NY hpc cluster, and might also works for hpc cluster in other schools, assuming your hpc is linux and is using Slurm. 

### conda init
First we need to login to the hpc. 
```
ssh netid@hpc.shanghai.nyu.edu
```
After this you should be on the login node of the hpc. Note the login node is different from a compute node, we will set up environment on the login node, then when we submit actual jobs, they are in fact run on the compute node. 

Note that on the hpc, students typically don't have admin privileges (which means you cannot install things that require `sudo`), so for some of the required system packages, we will not use the typical `sudo apt install` command, instead, we will use `module avail` to check if they are available, and then use `module load` to load them. If on your hpc cluster, a system package is not there, check with your hpc admin and ask them to help you. 

On the NYU Shanghai hpc (after you ssh, you get to the login node), first we want to set up conda correctly (typically need to do this for new accounts):
```
module load anaconda3
conda init bash
```
Now use Ctrl + D to logout, and then login again.

### set up MuJoCo

Now we are again on the hpc login node simply run this to load all required packages:
```
module load anaconda3 cuda/11.3.1
```

Now download MuJoCo files, on a linux machine, we put them under ~/.mujoco:
```
cd ~
mkdir ~/.mujoco
cd ~/.mujoco
curl -O https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
```

### set up conda environment
Then set up a conda virtualenv, and activate it (you can give it a different name, the env name does not matter)
```
conda create -y -n redq python=3.8
conda activate redq
```

Install Pytorch (skip this step if you don't need pytorch)
```
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

After you installed Pytorch, check if it works by running `python`, then `import torch` in the python interpreter. 
If Pytorch works, then either run `quit()` or Ctrl + D to exit the python interpreter. 

### set up `mujoco_py`

The next is to install MuJoCo, let's first run these 
```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
```
(it might be easier to do this step here. However, you can also set this up later when you test `import mujoco_py`)

Now we want the `.bashrc` file to take effect, we need to ctrl+D to logout, and then login again, after we logout and login, we need to reload all the modules , and then also activate the conda env again. (each time you login, the loaded modules and activated environment will be reset, but files on disk will persist) After you login to the hpc again:
```
module load anaconda3 cuda/11.3.1
conda activate redq
```

Now we can install `mujoco_py`. (You might want to know why do we need this? We already have the MuJoCo files, but they do not work directly with python, so `mujoco_py` is needed for us to use MuJoCo in python):
```
cd ~
mkdir rl_course
cd rl_course
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py/
pip install -e . --no-cache
```

Somewhere during installation, you might find that some libraries are missing (an error message might show up saying sth is missing, for example `GL/glew.h: No such file or directory #include <GL/glew.h>`), on the HPC, since we are not super user, we cannot install system libraries, but can use `module load` to load them. This command works on some machines: `module load glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1`. You can try this and then see if the error goes away. 

Now we want to test whether it works, run `python`, and then in the python interpreter, run `import mujoco_py`. The first time you run it, it will give a ton of log, if you can run it again and get no log, then it should be working. (Summary: try run `import mujoco_py` twice, if the second time you do it, you get no log and no error message, then it should be working). After this testing is complete, quit the python interpreter with either `quit()` or Ctrl + D. 

Note: if you got an error when import `mujoco_py`, sometimes the error will tell you to add some `export` text to a file (typically `~/.bashrc`) on your system, if you see that, then likely this is because you are installing on a system that configured things slightly differently from NYU Shanghai hpc cluster, in that case, just follow the error message to do whatever it tells you, then logout and login, and test it again. Check https://github.com/openai/mujoco-py for more info. 

### set up gym

Now we install OpenAI gym:
`pip install gym`

After this step, test gym by again run `python`, in the python interpreter, run:
```
import gym
e = gym.make('Ant-v2')
e.reset()
```
If you see a large numpy array (which is the initial state, or initial observation for the Ant-v2 environment), then gym is working. 

### set up REDQ
After this step, you can install REDQ. 
```
cd ~
cd rl_course
git clone https://github.com/watchernyu/REDQ.git
cd REDQ/
pip install -e .
```

### test on login node (sometimes things work on login but not on compute, we will first test login)
```
cd ~/rl_course/REDQ
python experiments/train_redq_sac.py --debug
```

### test on compute node (will be updated soon)
Now we test whether REDQ runs. We will first login to an interactive compute node (note a login node is not a compute node, don't do intensive computation on the login node.):
```
srun -p aquila --pty --mem  5000 -t 0-05:00 bash
```
And now don't forget we are in a new node and need to load modules and activate conda env:
```
module load anaconda3 cuda/11.3.1 
conda deactivate
conda activate redq 
```
Now test redq algorithm:
```
cd ~/rl_course/REDQ
python experiments/train_redq_sac.py --debug
```

<a name="acknowledgement"/> 

## other HPC issues
### missing patchelf
Simply install patchelf. Make sure your conda environment is activated, try this command (make sure you are inside your conda env): `conda install -c anaconda patchelf`.

If you see warnings, or a message telling you to update conda, ignore it, if it asks you whether you want to install, choose yes and wait for the installation to finish.

### quota exceeded
If you home quota is exceeded, you can contact the current HPC admin to extend your quota. Alternatively, you can install all require packages under `\scratch`, which has plenty of space (but your data under scratch will be removed if you don't use your account for too long). But you might need more python skills to do this correctly. 

## Acknowledgement

Our code for REDQ-SAC is partly based on the SAC implementation in OpenAI Spinup (https://github.com/openai/spinningup). The current code structure is inspired by the super clean TD3 source code by Scott Fujimoto (https://github.com/sfujim/TD3). 

