#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define THREADS_PER_BLOCK 1024

enum struct Parameter { ZERO, ONE, ID, RAND };

__managed__ int RAND_POSSIBLE_OUTCOME = 2;
__managed__ int NUM_PARAMETERS = 3;
__managed__ int NUM_RANDOM_PARAMETERS = 3;
__managed__ int NUM_SHOTS = 4;

__managed__ uint64_t STATE_COUNTER = 1;

__managed__ float DISTRIBUTION[2];

__managed__ Parameter **params;

typedef struct State {
  float a;
} State;

typedef struct BatchedTask {
  std::vector<std::vector<Parameter>> params;
  std::vector<State> states;
  uint64_t num_shots;

} BatchedTask;

__global__ void run(State *states, uint64_t num_shots, int param_idx) {

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t batch_idx = global_idx / num_shots;
  uint64_t shot_idx = global_idx % num_shots;

  if (threadIdx.x == 0) {
    printf("Waiting to be transformed by parameter %d\n", param_idx);
  }

  // Out of bounds
  if (global_idx >= num_shots * RAND_POSSIBLE_OUTCOME * NUM_RANDOM_PARAMETERS) {
    return;
  }

  // Transformation
  if (params[shot_idx][param_idx] == Parameter::ZERO) {

    if (global_idx < num_shots * STATE_COUNTER) {
      states[global_idx].a = 0.0;
      printf("Setting shot %lu in batch %lu to 0\n", shot_idx, batch_idx);
    }

  } else if (params[shot_idx][param_idx] == Parameter::ONE) {

    if (global_idx < num_shots * STATE_COUNTER) {
      states[global_idx].a = 1.0;
      printf("Setting shot %lu in batch %lu to 0\n", shot_idx, batch_idx);
    }
  } else if (params[shot_idx][param_idx] == Parameter::RAND) {

    if (global_idx < num_shots * STATE_COUNTER) {

      for (int i = 0; i < RAND_POSSIBLE_OUTCOME; i++) {
        states[global_idx + i * num_shots * STATE_COUNTER].a = DISTRIBUTION[i];
      }
    }

    STATE_COUNTER *= RAND_POSSIBLE_OUTCOME;

    __syncthreads();

    if (global_idx < num_shots * STATE_COUNTER) {
      printf("global_idx %lu, value: %f\n", global_idx, states[global_idx].a);
    }
  }
}

__global__ void reduce(State *states, uint64_t num_shots) {
  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t shot_idx = global_idx % num_shots;
  uint64_t batch_idx = global_idx / num_shots;

  extern __shared__ State per_shot_data[];

  if (global_idx >= STATE_COUNTER * num_shots) {
    return;
  }

  if (blockIdx.x == shot_idx) {
    for (int i = 0; i < RAND_POSSIBLE_OUTCOME ; i++) {
      per_shot_data[i] = states[global_idx + i * num_shots * STATE_COUNTER];
    }
  }

  __syncthreads();

  states[threadIdx.x] = per_shot_data[threadIdx.x];

  __syncthreads();

  if (threadIdx.x < STATE_COUNTER) {
    printf("Experiment %u, shot %lu: %f\n", blockIdx.x, shot_idx,
           states[threadIdx.x].a);
  }
}

__global__ void getCount(State *states, uint64_t num_shots) {

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t result;

  result = states[global_idx].a;

  extern __shared__ State counts[];

  if (threadIdx.x < RAND_POSSIBLE_OUTCOME) {
    counts[threadIdx.x].a = 0;
  } else {
    return;
  }

  __syncthreads();

  atomicAdd(&counts[result].a, 1);

  __syncthreads();

  if (threadIdx.x < RAND_POSSIBLE_OUTCOME) {
    states[threadIdx.x] = counts[threadIdx.x];
  }

  __syncthreads();
}

int main(int argc, char **argv) {

  // =================================================================================================================\
  // Setting up task

  for (int i = 0; i < RAND_POSSIBLE_OUTCOME; i++) {
    DISTRIBUTION[i] = i * 10;
    printf("Distribution %d: %f\n", i, DISTRIBUTION[i]);
  }

  BatchedTask task;
  task.num_shots = NUM_SHOTS;

  for (int i = 0; i < NUM_SHOTS; i++) {
    task.params.push_back({Parameter::RAND, Parameter::RAND, Parameter::RAND});
  }

  for (int i = 0; i < NUM_SHOTS; i++) {
    task.states.push_back({0.0});
  }

  State *states_ptr;

  printf("Number of shots: %lu\n", task.num_shots);
  printf("Number of parameters: %d\n", NUM_PARAMETERS);
  printf("Number of possible outcomes: %d\n", RAND_POSSIBLE_OUTCOME);

  // =================================================================================================================
  // Check for CUDA device

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

  // =================================================================================================================
  // Setting parameters for CUDA device
  printf("Preparing memory space\n");

  cudaDeviceReset();

  printf("Number of random parameters: %d\n", NUM_RANDOM_PARAMETERS);

  dim3 gridDim;

  cudaError_t status;

  //// dataOriginal = paramsOriginal + gridDim.x * THREADS_PER_BLOCK;
  //// printf("Start allocating memory for data at location: %f\n",
  /// dataOriginal);

  status = cudaMalloc((void **)&states_ptr, sizeof(State) * task.num_shots *
                                                NUM_RANDOM_PARAMETERS *
                                                RAND_POSSIBLE_OUTCOME);
  if (status != cudaSuccess) {
    printf("Error allocating memory for initial states - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for initial states\n");
  }

  gridDim.x = task.num_shots * NUM_RANDOM_PARAMETERS * RAND_POSSIBLE_OUTCOME /
              THREADS_PER_BLOCK;
  if (task.num_shots * NUM_RANDOM_PARAMETERS * RAND_POSSIBLE_OUTCOME %
          THREADS_PER_BLOCK !=
      0) {
    gridDim.x++;
  }
  // TODO: fix kernel
  printf("Grid dim: %d\n", gridDim.x);
  printf("Block dim: %d\n", THREADS_PER_BLOCK);

  cudaMallocManaged((void **)&params, sizeof(Parameter *) * task.num_shots);
  for (int i = 0; i < task.num_shots; i++) {
    cudaMallocManaged((void **)&params[i], sizeof(Parameter) * NUM_PARAMETERS);

    for (int j = 0; j < task.params[0].size(); j++) {
      params[i][j] = task.params[i][j];
      printf("Shot %d, parameter %d: %d\n", i, j, params[i][j]);
    }
  }

  // =================================================================================================================
  // Copy data to device

  // status = cudaMemcpy((void *)states_ptr, &(task.state),
  //                     sizeof(State) * task.num_shots * RAND_POSSIBLE_OUTCOME
  //                     *
  //                         NUM_RANDOM_PARAMETERS,
  //                     cudaMemcpyHostToDevice);
  // if (status != cudaSuccess) {
  //   printf("Error copying initial states to device - %s\n",
  //          cudaGetErrorString(status));
  //   exit(EXIT_FAILURE);
  // } else {
  //   printf("Copied initial states to device\n");
  // }

  printf("State counter initially: %lu\n", STATE_COUNTER);

  // =================================================================================================================
  // Running task

  printf("Running task\n");

  for (uint64_t i = 0; i < NUM_PARAMETERS; i++) {

    run<<<gridDim, THREADS_PER_BLOCK>>>(states_ptr, task.num_shots, i);

    cudaDeviceSynchronize();

    printf("Finish executing parameter %lu\n", i);

    // State temp[task.num_shots * STATE_COUNTER];
    // cudaMemcpy((void *)&temp, (void *)states_ptr,
    //            sizeof(State) * task.num_shots * STATE_COUNTER,
    //            cudaMemcpyDeviceToHost);

    // for (int i = 0; i < task.num_shots*STATE_COUNTER; i++) {
    //   size_t state_idx = i / task.num_shots;
    //   size_t shot_idx = i % task.num_shots;
    //   printf("Shot %lu, state %lu: %f\n", shot_idx, state_idx, temp[i].a);
    // }
  }

  printf("Start reducing\n");
  reduce<<<task.num_shots, THREADS_PER_BLOCK,
           sizeof(State) * THREADS_PER_BLOCK>>>(states_ptr, task.num_shots);

  cudaDeviceSynchronize();

  printf("Start gathering final counts\n");
  getCount<<<task.num_shots, THREADS_PER_BLOCK>>>(states_ptr, task.num_shots);

  cudaDeviceSynchronize();

  State final_results[task.num_shots * RAND_POSSIBLE_OUTCOME];

  cudaMemcpy((void *)&final_results, (void *)states_ptr,
             sizeof(State) * task.num_shots * RAND_POSSIBLE_OUTCOME,
             cudaMemcpyDeviceToHost);

  for (int s = 0; s < task.num_shots; s++) {
    printf("*******************************************************************"
           "*\n");
    for (int j = 0; j < RAND_POSSIBLE_OUTCOME; j++) {
      printf("Shot %d, outcome %d: %f\n", s, j,
             final_results[s * RAND_POSSIBLE_OUTCOME + j].a);
    }
    printf("*******************************************************************"
           "*\n");
  }

  // =================================================================================================================
  // Printing stats

  // =================================================================================================================
  // Benchmarking

  cudaFree(states_ptr);
}
