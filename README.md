## Please look at the savoir branch for the delta final project implementation (https://github.com/ashukla21/DRL_REDQ_Project/tree/savior)

## Data and reproducing figures in REDQ
We reproduced the figures the authors discussed in the paper, which can all be located under the the plot_utils folder. We used their dataset: https://drive.google.com/file/d/11mjDYCzp3T1MaICGruySrWd-akr-SVJp/view?usp=sharing, (Google Drive Link, ~80 MB)

## Steps that we followed to rectreate the results from the paper

First created a conda environment and activated it:
```
conda create -n redq python=3.6
conda activate redq 
```

Installed PyTorch (or you can follow the tutorial on PyTorch official website).
On Ubuntu (might also work on Windows but is not fully tested):
```
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
```
```
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch
```

OpenAI's gym (0.17.2):
```
git clone https://github.com/openai/gym.git
cd gym
git checkout b2727d6
pip install -e .
cd ..
```

mujoco_py (2.0.2.1): 
```
git clone https://github.com/openai/mujoco-py
cd mujoco-py
git checkout 379bb19
pip install -e . --no-cache
cd ..
```

Cloned the REDQ repo:
```
git clone https://github.com/watchernyu/REDQ.git
cd REDQ
pip install -e .
```

<a name="train-redq"/> 

## Trained a REDQ agent
```
python experiments/train_redq_sac.py
```
Used Georgia Tech's Pace H100 GPUs
