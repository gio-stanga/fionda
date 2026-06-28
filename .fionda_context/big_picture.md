# Fionda
Dynamic Scheduling of dense LLMs layers from host memory to VRAM over PCIe

## Goal
To enabel the local inference of dense LLMs that are bigger than the available VRAM using consumer grade NVIDIA GPUs while maintaining decent token/sec performances.

## Description of the Idea
To reach the goal we are going to for llama.cpp and rewrite (ora create if needed) the scheduler that transfers layers from the kost RAM to the device VRAM over PCIe.
The scheduler has to dynamically move layers from RAM to VRAM in this way: While layer N is being computed on the device, the scheduler starts loading layer N+1 in the device over PCIe. when N+1 is computing in the device the scheduler launches layer N+1 and takes back to the RAM layer N-1, which is no longer needed.
This way we leverage the latency iding of a layer being computed in the device while loading the following layer in the device in the meantime.

The data transfer is much slower than the computation of one layers, so we need to leverage techniques to make the data transfer faster, namely:
- using quantized models to linearly speedup transfer
- use speculative decoding to predict the following token and enable the possibility to process multiple tokens on the same layer while it is already in memory, increasing the compute time of the gpu and giving us more room to upload the following layer.

## Corrent State
At the moment the goal is to read and understand llama.cpp, in order to know what needs to be changes and/or created.
After that we will write the scheduler for NVIDIA GPUs.

## Strategy
llama.cpp is a big and complex codebase cosidering many edge cases. we are just gonna work for the most basic scenario, which is dense LLM, GGUF format from Hugging Face with the possibility to run models quiantized to INT8, and INT4.
We are already inside a clone of llama.cpp
