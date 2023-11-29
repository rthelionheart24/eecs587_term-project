#include <stdio.h>
#include <stdlib.h>

#include "include/backend.h"
#include "include/data.h"

#define THREADS_PER_BLOCK 1024

BatchedGPUTask::BatchedGPUTask(QCircuit *qc, unsigned int num_shots) {
  this->circuit = qc;
  this->num_shots = num_shots;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("No CUDA devices found\n");
    exit(EXIT_FAILURE);
  }

  cudaError_t err = cudaSetDevice(0);

  if (err != cudaSuccess) {
    printf("Error setting CUDA device - %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  size_t free_mem, total_mem;

  cudaError_t cuda_status = cudaMemGetInfo(&free_mem, &total_mem);
  if (cuda_status != cudaSuccess) {
    printf("Error getting CUDA memory info - %s\n",
           cudaGetErrorString(cuda_status));
    exit(EXIT_FAILURE);
  }

  printf("Free memory: %lu\n", free_mem);
  printf("Total memory: %lu\n", total_mem);
}

BatchedGPUTask::~BatchedGPUTask() {
  cudaError_t err = cudaDeviceReset();

  if (err != cudaSuccess) {
    printf("Error resetting CUDA device - %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

 __global__ void run(QCircuit *circuit) {
  return;
}

 void BatchedGPUTask::runWrapper() {

  printf("Running Simulation task\n");

  float *d_qubits;
  float *d_gates;
  float *d_bits;

  // Allocate memory on GPU
  cudaError_t status;

  status =
      cudaMalloc((void **)&d_qubits, sizeof(float) * this->circuit->num_qubits);

  if (status != cudaSuccess) {
    printf("Error allocating memory for qubits on GPU - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for qubits on GPU\n");
  }

  status =
      cudaMalloc((void **)&d_bits, sizeof(float) * this->circuit->num_bits);

  if (status != cudaSuccess) {
    printf("Error allocating memory for classical bits on GPU - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for classical bits on GPU\n");
  }

  status =
      cudaMalloc((void **)&d_gates, sizeof(float) * this->circuit->num_gates);

  if (status != cudaSuccess) {
    printf("Error allocating memory for gates on GPU - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for gates on GPU\n");
  }

  // Copy circuit to GPU
  status = cudaMemcpy(d_qubits, this->circuit->qubits.data(),
                      sizeof(float) * this->circuit->num_qubits,
                      cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_bits, this->circuit->bits.data(),
                      sizeof(float) * this->circuit->num_bits,
                      cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_gates, this->circuit->gates.data(),
                      sizeof(float) * this->circuit->num_gates,
                      cudaMemcpyHostToDevice);

  this->qubits_start_idx = 0;
  this->bits_start_idx = this->circuit->num_qubits;
  this->gates_start_idx = this->circuit->num_qubits + this->circuit->num_bits;

  if (status != cudaSuccess) {
    printf("Error copying circuit to GPU - %s\n", cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied circuit to GPU\n");
  }

  // Define grid and block dimensions
  dim3 gridDim;
  gridDim.x = 1;
  unsigned int size_circuit = this->circuit->num_qubits +
                              this->circuit->num_bits +
                              this->circuit->num_gates;

  gridDim.x = size_circuit / THREADS_PER_BLOCK;

  if (size_circuit % THREADS_PER_BLOCK != 0) {
    gridDim.x++;
  }

  printf("Grid dimensions: %d\n", gridDim.x);
  printf("Block dimensions: %d\n", THREADS_PER_BLOCK);

  // Run circuit on GPU

  run<<<gridDim, THREADS_PER_BLOCK>>>(this->circuit);

  cudaDeviceSynchronize();

  // Copy results back to CPU
  status = cudaMemcpy(this->circuit->qubits.data(), d_qubits,
                      sizeof(float) * this->circuit->num_qubits,
                      cudaMemcpyDeviceToHost);
  status = cudaMemcpy(this->circuit->bits.data(), d_bits, sizeof(float) * this->circuit->num_bits,
                      cudaMemcpyDeviceToHost);
  status = cudaMemcpy(this->circuit->gates.data(), d_gates,
                      sizeof(float) * this->circuit->num_gates , cudaMemcpyDeviceToHost);

  if (status != cudaSuccess) {
    printf("Error copying results back to CPU - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied results back to CPU\n");
  }

  cudaFree(d_qubits);
  cudaFree(d_bits);
  cudaFree(d_gates);
 }
