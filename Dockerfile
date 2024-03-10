#FROM nvidia/cuda:12.3.2-devel-ubuntu20.04
FROM ubuntu:20.04

WORKDIR /

RUN apt-get update && apt-get install wget -yq
RUN apt-get cmake && apt-get install ninja-build && apt-get clang
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/llvm-project-11.0.1.src.tar.xz
