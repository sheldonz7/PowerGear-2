
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
#FROM ubuntu:20.04

WORKDIR /



RUN apt-get update && apt-get install wget -yq
RUN apt-get install cmake -yq && apt-get install ninja-build -yq
RUN apt-get install -y --no-install-recommends ca-certificates cmake python subversion git
# replace llvm-11 with desired version
RUN apt-get install -y llvm-11
RUN apt-get install -y clang-11
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libedit-dev
RUN apt-get install -y p7zip-full
# needed for Python graphviz to work
RUN apt-get install -y graphviz

# install conda for Python env management
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh
