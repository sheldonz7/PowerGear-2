FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04
#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
#FROM ubuntu:20.04

WORKDIR /

RUN apt-get update && apt-get install wget -yq
RUN apt-get cmake && apt-get install ninja-build && apt-get clang
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/llvm-project-11.0.1.src.tar.xz
RUN tar -xvf llvm-project-11.0.1.src.tar.xz
RUN cd llvm-project-11.0.1.src
#RUN mkdir llvm-project-11.0.1.src/build
RUN cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld"
RUN ninja -C build check-llvm