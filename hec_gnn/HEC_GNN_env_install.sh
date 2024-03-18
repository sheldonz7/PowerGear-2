#!/usr/bin/env bash
# make sure command is : source HEC_GNN_env_install.sh

# install anaconda3.
# cd ~/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
# bash Anaconda3-2019.07-Linux-x86_64.sh


source ~/.bashrc
# export TORCH_CUDA_ARCH_LIST="7.0;7.5"   # v100: 7.0; 2080ti: 7.5; titan xp: 6.1

# # make sure system cuda version is the same with pytorch cuda
# # follow the instruction of PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# export PATH=/usr/local/cuda-11.3/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

conda create -n HEC_GNN python==3.8
conda activate HEC_GNN
# make sure pytorch version >=1.4.0
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorboard

# command to install pytorch geometric, please refer to the official website for latest installation.
#  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA=cu118
TORCH=2.0.1
#pip install torch-geometric==1.6.3
conda install pyg -c pyg

# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TOECH}+${CUDA}.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# if error,please try:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.7.0+cu110.html

pip install requests

# install useful modules
pip install tqdm

# additional package required for ogb experiments
pip install ogb
### check the version of ogb installed, if it is not the latest
# python -c "import ogb; print(ogb.__version__)"
# please update the version by running
# pip install -U ogb

# additional package required for dgl implementation
# pip install dgl-cu102
