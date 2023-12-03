#include <stdio.h>
#include <stdlib.h>

#include "include/backend.h"
#include "include/data.h"

template <typename D, typename P>
BatchedGPUTask::BatchedGPUTask(Task<D, P> *task, unsigned int num_shots) {
  this->task = task;
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

void prepare(){
  printf("Preparing memory space\n");

  gridDim.x = 0;

  // =================================================================================================================
  // Allocate memory on device

  cudaError_t status;

  // printf("Start allocating for params at location: %f\n", d_params);
  status = cudaMalloc((void **)&this->params_ptr,
                      sizeof(P) * this->task->params.size());
  if (status != cudaSuccess) {
    printf("Error allocating memory for params - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for params\n");
  }

  gridDim.x += this->task->params.size() / THREADS_PER_BLOCK;
  if (this->task->params.size() % THREADS_PER_BLOCK != 0) {
    gridDim.x++;
  }


  //// dataOriginal = paramsOriginal + gridDim.x * THREADS_PER_BLOCK;
  //// printf("Start allocating memory for data at location: %f\n", dataOriginal);

  status = cudaMalloc((void **)&data_ptr,
                      sizeof(T) * this->task->data.size());
  if (status != cudaSuccess) {
    printf("Error allocating memory for data - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for data\n");
  }

  gridDim.x += this->task->data.size() / THREADS_PER_BLOCK;
  if (this->task->data.size() % THREADS_PER_BLOCK != 0) {
    gridDim.x++;
  }

  printf("Grid dim: %d\n", gridDim.x);
  printf("Block dim: %d\n", THREADS_PER_BLOCK);

  this->data_idx = this->numBlocksParams * THREADS_PER_BLOCK;


  // =================================================================================================================
  // Copy data to device

  status =
      cudaMemcpy(d_params, this->task->params,
                 sizeof(P) * this->task->data.size(), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    printf("Error copying params to device - %s\n", cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied params of size %d to device\n", this->task->params.size());
  }

  status =
      cudaMemcpy(d_data, this->task->data,
                 sizeof(T) * this->task->params.size(), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    printf("Error copying data to device - %s\n", cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied data of length %d to device\n", this->task->data.size());
  }


}


__device__ transform(){

}


template <typename D, typename P>
 __global__ void run(BatchedGPUTask<typename D, typename P> *task) {
  
   printf("Running task\n");

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // tid falls into the parameter section
  if (tid < this->data_idx) {
    return;
  }

  // tid out of bounds
  if (tid > task->data_idx + this->statesCounter) {
    return;
  }

  // tid falls into the data section
  transform()


}
template <typename D, typename P>
 void BatchedGPUTask::runWrapper() {

  printf("Running task wrapper\n");

  prepare();


  run<<<gridDim, THREADS_PER_BLOCK>>>(this);

 
 }
