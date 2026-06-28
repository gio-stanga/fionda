# Fionda
Dynamic Scheduling of dense LLMs layers from host memory to VRAM over PCIe

## Goal
To enabel the local inference of dense LLMs that are bigger than the available VRAM using consumer grade NVIDIA GPUs while maintaining decent token/sec performances.

## Description of the Idea
To reach the goal we are going to fork llama.cpp and rewrite (or create if needed) the scheduler that transfers layers from the host RAM to the device VRAM over PCIe.
The scheduler has to dynamically move layers from RAM to VRAM in this way: while layer N is being computed on the device, the scheduler starts loading layer N+1 in the device over PCIe. When N is computing in the device the scheduler launches layer N+1 and takes back to the RAM layer N-1, which is no longer needed.
This way we leverage the latency hiding of a layer being computed in the device while loading the following layer in the device in the meantime.

The data transfer is much slower than the computation of one layer, so we need to leverage techniques to make the data transfer faster, namely:
- using quantized models to linearly speedup transfer
- use speculative decoding to predict the following token and enable the possibility to process multiple tokens on the same layer while it is already in memory, increasing the compute time of the gpu and giving us more room to upload the following layer.

## Current State
At the moment the goal is to read and understand llama.cpp, in order to know what needs to be changed and/or created.
After that we will write the scheduler for NVIDIA GPUs.

## Strategy
llama.cpp is a big and complex codebase cosidering many edge cases. We are just gonna work for the most basic scenario, which is dense LLM, GGUF format from Hugging Face with the possibility to run models quiantized to INT8, and INT4.
We are already inside a clone of llama.cpp
