
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "include/data.h"

// ! Need to change the way we parallelize the problem
// ! Approach 1: Different experiments of the same states are put together
// ? Approach 2: One block per experiment

__managed__ uint64_t STATE_COUNTER = 1;

__managed__ float DISTRIBUTION[5];
__managed__ int RAND_POSSIBLE_OUTCOME = 5;
__managed__ int NUM_PARAMETERS = -1;
__managed__ int NUM_RANDOM_PARAMETERS = 5;
__managed__ int NUM_SHOTS = 2;

__managed__ Parameter **params;

__global__ void run(State *states, uint64_t num_shots,
                    uint64_t num_blocks_per_shot, int param_idx) {

  extern __shared__ State per_shot_data[];

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t shot_idx = blockIdx.x / num_blocks_per_shot;

  // Transformation
  if (params[shot_idx][param_idx] == Parameter::X_OP) {

    if (blockIdx.x == shot_idx && threadIdx.x < STATE_COUNTER) {
      per_shot_data[threadIdx.x].a = 0.0;
      // printf("DEVICE: Setting state %u in block %u to value: %f\n",
      // threadIdx.x,
      //  blockIdx.x, per_shot_data[threadIdx.x].a);
    }

    __syncthreads();

  } else if (params[shot_idx][param_idx] == Parameter::Y_OP) {

    if (blockIdx.x == shot_idx && threadIdx.x < STATE_COUNTER) {
      per_shot_data[threadIdx.x].a = 1.0;
      // printf("DEVICE: Setting state %u in block %u to value: %f\n",
      // threadIdx.x,
      //  blockIdx.x, per_shot_data[threadIdx.x].a);
    }

    __syncthreads();

  } else if (params[shot_idx][param_idx] == Parameter::Z_OP) {

    if (blockIdx.x == shot_idx && threadIdx.x < STATE_COUNTER) {
      per_shot_data[threadIdx.x].a = 3.0;
      // printf("DEVICE: Setting state %u in block %u to value: %f\n",
      // threadIdx.x,
      //  blockIdx.x, per_shot_data[threadIdx.x].a);
    }

    __syncthreads();
  } else if (params[shot_idx][param_idx] == Parameter::ID) {
    // if (threadIdx.x < STATE_COUNTER) {
    //   printf("DEVICE: state %u in block %u remains identical\n",
    //   threadIdx.x, blockIdx.x);
    // }
  }

  // HACK: This is a hack to make sure that the states are not overwritten
  states[global_idx] = per_shot_data[threadIdx.x];

  __syncthreads();

  // printf("DEVICE: shot: %u, state, %u: value: %f\n", blockIdx.x, threadIdx.x,
  //  states[threadIdx.x].a);
}

__global__ void run_random(State *states, uint64_t num_shots,
                           uint64_t num_blocks_per_shot, int param_idx) {
  extern __shared__ State per_shot_data[];

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t shot_idx = blockIdx.x / num_blocks_per_shot;

  if (blockIdx.x == shot_idx && threadIdx.x < STATE_COUNTER) {

    for (int i = 0; i < RAND_POSSIBLE_OUTCOME; i++) {
      int new_idx = threadIdx.x + i * STATE_COUNTER;
      if (new_idx >= 1024) {
        new_idx = global_idx + i * STATE_COUNTER;
      }

      per_shot_data[new_idx].a = DISTRIBUTION[i];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    STATE_COUNTER *= RAND_POSSIBLE_OUTCOME;
  }

  // HACK: This is a hack to make sure that the states are not overwritten
  states[global_idx] = per_shot_data[threadIdx.x];

  __syncthreads();

  // printf("DEVICE: shot: %u, state, %u: value: %f\n", blockIdx.x, threadIdx.x,
  //  states[threadIdx.x].a);
}

void memoryInfo() {
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  printf("Free memory: %lu\n", free_mem);
  printf("Total memory: %lu\n", total_mem);
}

int main(int argc, char **argv) {

  // =================================================================================================================\
  // Immitating user setting up a batched task

  for (int i = 0; i < RAND_POSSIBLE_OUTCOME; i++) {
    DISTRIBUTION[i] = i * 0.1;
    printf("Distribution %d: %f\n", i, DISTRIBUTION[i]);
  }

  BatchedTask task;
  task.num_shots = NUM_SHOTS;

  task.params.push_back(
      {Parameter::ID, Parameter::RAND_OP, Parameter::Y_OP, Parameter::X_OP,
       Parameter::RAND_OP, Parameter::RAND_OP, Parameter::X_OP,
       Parameter::RAND_OP, Parameter::RAND_OP, Parameter::X_OP});
  task.params.push_back({Parameter::ID, Parameter::RAND_OP, Parameter::X_OP,
                         Parameter::Y_OP, Parameter::RAND_OP, Parameter::X_OP,
                         Parameter::RAND_OP, Parameter::RAND_OP,
                         Parameter::RAND_OP, Parameter::Y_OP});
  // task.params.push_back({Parameter::ID, Parameter::RAND_OP, Parameter::X_OP,
  //                        Parameter::Y_OP, Parameter::RAND_OP,
  //                        Parameter::X_OP, Parameter::RAND_OP,
  //                        Parameter::RAND_OP, Parameter::RAND_OP,
  //                        Parameter::Z_OP});
  // task.params.push_back(
  //     {Parameter::ID, Parameter::RAND_OP, Parameter::X_OP,
  //     Parameter::RAND_OP,
  //      Parameter::X_OP, Parameter::RAND_OP, Parameter::X_OP,
  //      Parameter::RAND_OP, Parameter::RAND_OP, Parameter::ID});

  for (int i = 0; i < task.num_shots; i++) {
    task.states.push_back({0.0});
  }

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

  printf("Before allocating, memory info:\n");
  memoryInfo();

  // =================================================================================================================
  // Setting parameters for CUDA device

  printf("Preparing memory space\n");

  for (int e = 0; e < task.num_shots; e++) {
    NUM_PARAMETERS = std::max(NUM_PARAMETERS, (int)task.params[e].size());
  }

  for (int e = 0; e < task.num_shots; e++) {
    for (int i = task.params[e].size(); i < NUM_PARAMETERS; i++) {
      task.params[e].push_back(Parameter::ID);
    }
  }

  printf("Number of shots: %lu\n", task.num_shots);
  printf("Number of parameters: %d\n", NUM_PARAMETERS);
  printf("Number of random parameters: %d\n", NUM_RANDOM_PARAMETERS);
  printf("Number of possible outcomes: %d\n", RAND_POSSIBLE_OUTCOME);

  cudaDeviceReset();

  cudaError_t status;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // ! Start the timer
  cudaEventRecord(start);

  size_t MAXIMUM =
      static_cast<size_t>(pow(RAND_POSSIBLE_OUTCOME, NUM_RANDOM_PARAMETERS));

  State *states_ptr;
  int num_blocks_per_shot = MAXIMUM / 1024;
  if (MAXIMUM % 1024 != 0) {
    num_blocks_per_shot++;
  }

  // HACK Allocation
  status = cudaMalloc((void **)&states_ptr, sizeof(State) * task.num_shots *
                                                num_blocks_per_shot * 1024);
  if (status != cudaSuccess) {
    printf("Error allocating memory for initial states - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for initial states\n");
  }

  printf("Grid dim: %lu\n", task.num_shots * num_blocks_per_shot);
  printf("Block dim: %d\n", 1024);

  cudaMallocManaged((void **)&params, sizeof(Parameter *) * task.num_shots);
  for (int i = 0; i < task.num_shots; i++) {
    cudaMallocManaged((void **)&params[i], sizeof(Parameter) * NUM_PARAMETERS);

    for (int j = 0; j < task.params[0].size(); j++) {
      params[i][j] = task.params[i][j];
      // printf("Shot %d, parameter %d: %d\n", i, j, params[i][j]);
    }
  }

  printf("After allocating, memory info:\n");
  memoryInfo();

  // =================================================================================================================
  // Copy data to device

  State d_states[task.num_shots][MAXIMUM];

  for (int i = 0; i < task.num_shots; i++) {
    for (int j = 0; j < MAXIMUM; j++) {
      d_states[i][j].a = 0.0;
    }
  }

  status = cudaMemcpy((void *)states_ptr, &d_states,
                      MAXIMUM * task.num_shots * sizeof(State),
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

  int iter_param = 0;

  while (iter_param < NUM_PARAMETERS) {

    printf("Start executing parameter %d\n", iter_param);
    printf("STATE_COUNTER: %lu\n", STATE_COUNTER);


    bool all_rand_op = true;
    bool has_rand_op = false;

    std::vector<int> rand_op_idx;
    std::vector<int> non_rand_idx;

    for (int e = 0; e < task.num_shots; e++) {
      for (int p = 0; p < NUM_PARAMETERS; p++) {
        printf("Shot %d, parameter %d: %d\n", e, p, params[e][p]);
      }
    }

    for (int e = 0; e < task.num_shots; e++) {
      if (params[e][iter_param] != Parameter::RAND_OP) {
        all_rand_op = false;
        break;
      }
    }

    for (int e = 0; e < task.num_shots; e++) {

      if (params[e][iter_param] == Parameter::RAND_OP) {
        rand_op_idx.push_back(e);
        has_rand_op = true;
        // printf("Random operation found at index %d\n", e);
      } else {
        non_rand_idx.push_back(e);
        // printf("Non-random operation found at index %d\n", e);
      }
    }

    // HACK: To add padding ID gates to all shots that encounter a random
    // operation
    if (has_rand_op && !all_rand_op) {

      for (int s = 0; s < rand_op_idx.size(); s++) {
        int idx = rand_op_idx[s];
        Parameter *new_params;
        cudaMallocManaged((void **)&new_params,
                          sizeof(Parameter) * (NUM_PARAMETERS + 1));

        for (int i = 0; i < NUM_PARAMETERS + 1; ++i) {
          if (i < iter_param) {
            // Copy elements before the index
            new_params[i] = params[idx][i];
          } else if (i > iter_param) {
            // Shift elements after the index
            new_params[i] = params[idx][i - 1];
          }
        }

        new_params[iter_param] = Parameter::ID;
        cudaFree(params[idx]);
        params[idx] = new_params;
      }

      for (int s = 0; s < non_rand_idx.size(); s++) {
        int idx = non_rand_idx[s];
        Parameter *new_params;
        cudaMallocManaged((void **)&new_params,
                          sizeof(Parameter) * (NUM_PARAMETERS + 1));

        for (int i = 0; i < NUM_PARAMETERS; ++i) {
          // Copy elements before the index
          new_params[i] = params[idx][i];
        }

        new_params[NUM_PARAMETERS] = Parameter::ID;

        cudaFree(params[idx]);
        params[idx] = new_params;
      }

      NUM_PARAMETERS++;

      for (int e = 0; e < task.num_shots; e++) {
        for (int p = 0; p < NUM_PARAMETERS; p++) {
          // printf("Shot %d, parameter %d: %d\n", e, p, params[e][p]);
        }
      }
    }

    if (all_rand_op){
      run_random<<<task.num_shots * num_blocks_per_shot, 1024, sizeof(State) * 1024>>>(
        states_ptr, task.num_shots, num_blocks_per_shot, iter_param);
    }
    else{
    run<<<task.num_shots * num_blocks_per_shot, 1024, sizeof(State) * 1024>>>(
        states_ptr, task.num_shots, num_blocks_per_shot, iter_param);
    }

    cudaDeviceSynchronize();

    printf("STATE_COUNTER: %lu\n", STATE_COUNTER);
    printf("Finish executing parameter %d\n", iter_param);

    iter_param++;
  }

  printf("Start reducing\n");

  cudaDeviceSynchronize();

  cudaMemcpy((void *)d_states, (void *)states_ptr,
             sizeof(State) * MAXIMUM * task.num_shots, cudaMemcpyDeviceToHost);

  std::vector<std::unordered_map<float, uint64_t>> stats(task.num_shots);

  for (int e = 0; e < task.num_shots; e++) {
    for (int i = 0; i < MAXIMUM; i++) {
      // printf("HOST: Shot %d, state %d: %f\n", e, i, d_states[e][i].a);
      stats[e][d_states[e][i].a]++;
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // =================================================================================================================
  // Printing stats

  printf("Time taken: %f\n", elapsedTime);
  for (int i = 0; i < task.num_shots; i++) {
    printf("***********************************************\n");
    printf("Statistics for shot %d\n", i);
    for (auto &pair : stats[i]) {
      printf("Value: %f, frequency: %f\n", pair.first,
             (float)pair.second / STATE_COUNTER);
    }
    printf("***********************************************\n");
  }

  // =================================================================================================================
  // Benchmarking

  cudaFree(states_ptr);
  cudaFree(params);
}
