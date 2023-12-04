# Batched Quantum Circuit Simulation on GPUs

This project is a CUDA-based quantum simulation that performs batched quantum experiments on a GPU in parallel. It is designed to simulate the bahevior of multiple potentially different quantum circuit.


## Introduction

Quantum computers are devices that can perform quantum computations by exploting the quantum mechanical phenomena of superposition and entanglement. Quantum computers are expected to be able to solve certain problems much faster than classical computers. However, the development of quantum computers is still in its infancy. Currently, quantum computers are still too small to be useful for practical applications. Therefore, quantum simulation is an important tool for studying quantum algorithms and quantum computers.


### Status-quo of quantum simulation

Currently, there are a few ways to simulate a quantum system such as state vector, tensor network, density matrix, etc. Nevertheless, most methods use the same model: the operation of the quantum system is represented by an initial state running through a series of quantum operators. Since this is similar to the circuit model in classical computers, where an electron runs through a series of logic gates, we can use the circuit model to simulate quantum systems. As shown by figure 1, each circuit will run through a series of quantum operators, and the result will be the final state of the quantum system. THis project will also use the circuit model to simulate quantum systems.

![Alt text](figure-1.png)
*Figure 1: The circuit model of quantum simulation*

The state of a quantum system can be represented by a vector of complex numbers, a tensor node, a density matrix, etc, depending on the simulation methods. Nevertheless, no matter what method is used, the quantum states and operations on them are in a Hilbert space and can be represented by a matrix. Therefore, it is possible to design a simplified yet generalized model for quantum simulation by abstracting the state, no matter the data size of data structure, into a value, and abstracting the quantum operator into a function that transforms the state. As shown by figure 2, the quantum circuit model can be generalized as a series of quantum operators that transform the state of the quantum system.

![Alt text](figure-2.png)
*Figure 2: Generalized model of quantum circuit*

Many frameworks adapt the generalized circuit model to simulate quantum systems. For example, QuEST, Qiskit, and Cuquantum use the circuit model to simulate quantum systems. These frameworks supports a variety of architectures including GPU. However, the GPU support is limited to single-shot quantum experiments, meaning that for each run of the quantum circuit, the framework will launch a new GPU kernel to simulate the quantum circuit. This is inefficient because the overhead of launching a GPU kernel is high. Therefore, this project aims to improve the efficiency of quantum simulation on GPUs by performing batched quantum experiments on GPUs.


## Approach

![Alt text](figure-3.png)
*Figure 3: Batched generalized model of quantum circuit*

As shown by figure 3, we can group multiple quantum circuit into a batched GPU task. Each quantum circuit will run through a series of quantum operators, and the result will be the final state of the quantum system. The batched quantum simulation will run multiple quantum circuits in parallel on a GPU. This will reduce the overhead of launching a GPU kernel and improve the efficiency of quantum simulation on GPUs. 


However, different quantum circuit can be group together into a batched task. These circuits can vary in the number of quantum operators. Therefore, it is challenging to design a batched quantum simulation that can handle different quantum circuits. Furthermore, besides the definite quantum operators that are known before the simulation, there are also quantum operators that are determined during the simulation. For example, the measurement operator is determined by the measurement result of the previous quantum operator, and operators that causes random changes to the value is also present to model sent noise inside the quantum system. Therefore, it is challenging to design a batched quantum simulation that can handle quantum operators that are determined in runtime.

Another challenge is how to implement shot-branching. Shot-branching is a technique that allows the state to branch into multiple states, each of which is a copy of the original state, when a non-deterministic operator such as measurement or noise operator is encountered. This is useful for simulating quantum systems that are entangled with each other. However, it is challenging to implement shot-branching in a batched quantum simulation because the number of states can vary for each quantum circuit in the batched task depending on how many non-deterministic operations have been executed at a given moment. This is the main challenge of this project. 

Given the two major challenges, we need to think carefully about how to sychronize

### Design

One way to parallelize the problem is to allocate one block for each quantum circuit in the batched task; we will refer to it by "shot" for the rest of the report. There are a few constants that need to be set prior to the batched task being run, such as the number of shots, the number of quantum operators, etc. Each thread  


On the other hand, simulating a well-aligned batched quantum circuit is not difficult. By well-aligned, we mean that the number quantum operators is identical for each quantum circuit in the batched task. Furthermore, for each 




# References:

**Comparison between different frameworks:**
https://www.cirrus.ac.uk/news/2023/06/12/quantum.html

**Cuquantum official website:**
https://developer.nvidia.com/cuquantum-sdk

**QuEST:**
https://github.com/QuEST-Kit/QuEST/tree/master

**Paper on cache-blocking:**
https://arxiv.org/abs/2102.02957

**Quantum Computer Simulation at Warp Speed: Assessing the Impact of GPU Acceleration:**
https://arxiv.org/pdf/2307.14860.pdf

**Efficient techniques to GPU Accelerations of Multi-Shot Quantum Computing Simulations:**
https://arxiv.org/pdf/2308.03399.pdf
