---
title: Cuda Programming
author: Cheng Luo
date: 2024-01-13 19:40:00 +1100
categories: [Study]
tags: [Technique]
pin: true
math: true
---


##Core Concepts:
- **Host**: CPU and its memory
- **Device**: GPU and its memory


##Procedures
- Allocate the memory of host and initialize the data
- Allocate the memory of device and copy data from host to device
- Use the kernal functions of CUDA and complete computations on device
- Copy computed results from device to host
- Free the memory on both host and device 


** using the CUDA kernal functions to execute parallel computing

__global__ declares kernal
<<<grid, block>>> defines the number of executing threads
Each thread has a thread ID (threadIdx)


---