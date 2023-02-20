# cudagl with miniconda and python 3.8, pytorch, mujoco and gym v2 envs, REDQ

FROM nvidia/cudagl:11.0-base-ubuntu18.04
WORKDIR /workspace
ENV HOME=/workspace
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# idea: start with a nvidia docker with gl support (guess this one also has cuda?)
# then install miniconda, borrowing docker command from miniconda's Dockerfile (https://hub.docker.com/r/continuumio/anaconda/dockerfile/)
# need to make sure the miniconda python version is what we need (https://docs.conda.io/en/latest/miniconda.html for the right version)
# then install other dependencies we need

# nvidia GPG key alternative fix (https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772)
# sudo apt-key del 7fa2af80
# wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
# sudo dpkg -i cuda-keyring_1.0-1_all.deb

RUN \
    # Update nvidia GPG key
    rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg \
    #    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    #    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    #    dpkg -i tini.deb && \
    #    rm tini.deb && \
    && apt-get clean

# Install some basic utilities
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:redislabs/redis && apt-get update \
    && apt-get install -y sudo ssh libx11-6 gcc iputils-ping \
    libxrender-dev graphviz tmux htop build-essential wget cmake libgl1-mesa-glx redis \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y zlib1g zlib1g-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libglew2.0
    #    && ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
# ---------- now we should have all major dependencies ------------

# --------- now we have cudagl + python38 ---------
RUN pip install  --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir scikit-learn pandas imageio

# adroit env does not work with newer version of mujoco, so we have to use the old version...
# setting MUJOCO_GL to egl is needed if run on headless machine
ENV MUJOCO_GL=egl
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/workspace/.mujoco/mujoco210/bin
RUN mkdir -p /workspace/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz \
    && tar -xvzf mujoco210.tar.gz -C /workspace/.mujoco \
    && rm mujoco210.tar.gz

# RL env: get mujoco and gym (v2 version environments)
RUN pip install --no-cache-dir patchelf==0.17.2.0 mujoco-py==2.1.2.14 gym==0.21.0

# first time import mujoco_py will take some time to run, do it here to avoid the wait later
RUN echo "import mujoco_py" >> /workspace/import_mujoco.py \
    && python /workspace/import_mujoco.py

# RL algorithm: install REDQ. If you don't want to use REDQ, simply remove these lines
RUN cd /workspace/ \
    && git clone https://github.com/watchernyu/REDQ.git \
    && cd REDQ \
    && git checkout ac840198f143d10bb22425ed2105a49d01b383fa \
    && pip install -e .

CMD [ "/bin/bash" ]

# # build docker container:
# docker build . -t name:tag

# # example docker command to run interactive container, enable gpu, and remove it when shutdown:
# docker run -it --rm --gpus all name:tag
