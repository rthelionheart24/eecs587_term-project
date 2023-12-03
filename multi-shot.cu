
#include <cstddef>
#include <cstdlib>
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
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// ! Need to change the way we parallelize the problem
// ! Approach 1: Different experiments of the same states are put together
// ? Approach 2: One block per experiment

#define THREADS_PER_BLOCK 1024

enum struct Parameter { ZERO, ONE, ID, RAND };

__managed__ int RAND_POSSIBLE_OUTCOME = 2;
__managed__ int NUM_PARAMETERS = 3;
__managed__ int NUM_RANDOM_PARAMETERS = 2;
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

  extern __shared__ State per_shot_data[];

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t batch_idx = global_idx / num_shots;
  uint64_t shot_idx = global_idx % num_shots;

  // Transformation
  if (params[shot_idx][param_idx] == Parameter::ZERO) {

    if (threadIdx.x < STATE_COUNTER) {
      per_shot_data[threadIdx.x].a = 0.0;
      printf("DEVICE: Setting state %u in block %u to value: %f\n", threadIdx.x,
             blockIdx.x, per_shot_data[threadIdx.x].a);
    }

    __syncthreads();

  } else if (params[shot_idx][param_idx] == Parameter::ONE) {

    if (threadIdx.x < STATE_COUNTER) {
      per_shot_data[threadIdx.x].a = 1.0;
      printf("DEVICE: Setting state %u in block %u to value: %f\n", threadIdx.x,
             blockIdx.x, per_shot_data[threadIdx.x].a);
    }

    __syncthreads();

  } else if (params[shot_idx][param_idx] == Parameter::RAND) {

    if (threadIdx.x < STATE_COUNTER) {

      for (int i = 0; i < RAND_POSSIBLE_OUTCOME; i++) {
        per_shot_data[threadIdx.x + i * STATE_COUNTER].a = DISTRIBUTION[i];
      }
    }

    STATE_COUNTER *= RAND_POSSIBLE_OUTCOME;

    __syncthreads();

    if (threadIdx.x < STATE_COUNTER) {
      printf("DEVICE: Setting state %u in block %u to random value: %f\n",
             threadIdx.x, blockIdx.x, per_shot_data[threadIdx.x].a);
    }
  }
  __syncthreads();

  states[threadIdx.x] = per_shot_data[threadIdx.x];

  __syncthreads();

  printf("DEVICE: shot: %u, state, %u: value: %f\n", blockIdx.x, threadIdx.x,
         states[threadIdx.x].a);
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
    task.params.push_back({Parameter::ONE, Parameter::RAND, Parameter::RAND});
  }

  for (int i = 0; i < NUM_SHOTS; i++) {
    task.states.push_back({0.0});
  }

  State *states_ptr;

  printf("Number of shots: %lu\n", task.num_shots);
  printf("Number of parameters: %d\n", NUM_PARAMETERS);
  printf("Number of random parameters: %d\n", NUM_RANDOM_PARAMETERS);
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

  cudaError_t status;

  size_t MAXIMUM =
      static_cast<size_t>(pow(RAND_POSSIBLE_OUTCOME, NUM_RANDOM_PARAMETERS));

      //// dataOriginal = paramsOriginal + gridDim.x * THREADS_PER_BLOCK;
      //// printf("Start allocating memory for data at location: %f\n",
      /// dataOriginal);

      // HACK Allocation
      status =
          cudaMalloc((void **)&states_ptr, sizeof(State) * task.num_shots * MAXIMUM);
  if (status != cudaSuccess) {
    printf("Error allocating memory for initial states - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for initial states\n");
  }

  printf("Grid dim: %lu\n", task.num_shots);
  printf("Block dim: %zu\n", MAXIMUM);

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

  State d_states[task.num_shots][MAXIMUM];

  for (int i = 0; i < task.num_shots; i++) {
    for (int j = 0; j < RAND_POSSIBLE_OUTCOME * NUM_RANDOM_PARAMETERS; j++) {
      d_states[i][j].a = 0.0;
    }
  }

  status = cudaMemcpy((void *)states_ptr, &d_states, MAXIMUM * sizeof(State),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    printf("Error copying initial states to device - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied initial states to device\n");
  }

  // =================================================================================================================
  // Running task

  printf("Running task\n");

  State *results;

  for (uint64_t i = 0; i < NUM_PARAMETERS; i++) {

    run<<<task.num_shots,
          static_cast<int>(pow(RAND_POSSIBLE_OUTCOME, NUM_RANDOM_PARAMETERS)),
          sizeof(State) * STATE_COUNTER>>>(states_ptr, task.num_shots, i);

    cudaDeviceSynchronize();

    results = (State *)malloc(sizeof(State) * task.num_shots * STATE_COUNTER);

    cudaMemcpy((void *)results, (void *)states_ptr,
               sizeof(State) * task.num_shots * STATE_COUNTER,
               cudaMemcpyDeviceToHost);

    for (int e = 0; e < task.num_shots; e++) {
      for (int i = 0; i < STATE_COUNTER; i++) {
        printf("HOST: Shot %d, state %d: %f\n", e, i,
               results[STATE_COUNTER * e + i].a);
      }
    }

    free(results);

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

  // reduce<<<task.num_shots, STATE_COUNTER, sizeof(State) * STATE_COUNTER>>>(
  // states_ptr, task.num_shots);

  cudaDeviceSynchronize();

  printf("STATE_COUNTER: %lu\n", STATE_COUNTER);

  cudaMemcpy((void *)results, (void *)states_ptr,
             sizeof(State) * STATE_COUNTER * task.num_shots,
             cudaMemcpyDeviceToHost);

  // for (int e = 0; e < task.num_shots; e++){
  //   for (int i = 0; i < STATE_COUNTER; i++){
  //     printf("Value: %f, count: %d\n", states_ptr[e * STATE_COUNTER + i].a,
  //     counts[i]);
  //   }
  // }

  // =================================================================================================================
  // Printing stats

  // for (int i = 0; i < task.num_shots; i++) {
  //   printf("***********************************************\n");
  //   printf("Statistics for shot %d\n", i);
  //   for (auto &pair : stats[i]) {
  //     printf("Value: %f, frequency: %f\n", pair.first,
  //            (float)pair.second / STATE_COUNTER);
  //   }
  //   printf("***********************************************\n");
  // }

  // =================================================================================================================
  // Benchmarking

  cudaFree(states_ptr);
}
