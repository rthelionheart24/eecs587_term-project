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

__managed__ float DISTRIBUTION[40];
__managed__ int RAND_POSSIBLE_OUTCOME = 40;
__managed__ int NUM_PARAMETERS = -1;
__managed__ int NUM_RANDOM_PARAMETERS = 5;
__managed__ int NUM_SHOTS = 4;

__managed__ Parameter **params;

__global__ void run(State *states, uint64_t num_shots,
                    uint64_t num_blocks_per_shot, int param_idx) {

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t shot_idx = blockIdx.x / num_blocks_per_shot;
  uint64_t state_idx = global_idx % (num_blocks_per_shot * 1024);

  if (state_idx >= STATE_COUNTER)
    return;

  // Transformation
  if (params[shot_idx][param_idx] == Parameter::X_OP) {

    states[global_idx].a = 1.0;
    // printf("DEVICE: Setting state %u in block %u to value: %f\n",
    // threadIdx.x,
    //  blockIdx.x, per_shot_data[threadIdx.x].a);
  }
  if (params[shot_idx][param_idx] == Parameter::Y_OP) {

    states[global_idx].a = 2.0;
    // printf("DEVICE: Setting state %u in block %u to value: %f\n",
    // threadIdx.x,
    //  blockIdx.x, per_shot_data[threadIdx.x].a);
  }
  if (params[shot_idx][param_idx] == Parameter::Z_OP) {

    states[global_idx].a = 3.0;
    // printf("DEVICE: Setting state %u in block %u to value: %f\n",
    // threadIdx.x,
    //  blockIdx.x, per_shot_data[threadIdx.x].a);
  }
  if (params[shot_idx][param_idx] == Parameter::ID) {
    // printf("ID\n");
  }
}

__global__ void run_random(State *states, uint64_t num_shots,
                           uint64_t num_blocks_per_shot, int param_idx) {

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t shot_idx = blockIdx.x / num_blocks_per_shot;
  uint64_t state_idx = global_idx - shot_idx * num_blocks_per_shot * 1024;

  // printf("STATE_COUNTER: %lu\n", STATE_COUNTER);

  if (state_idx >= STATE_COUNTER) {
    // printf("state_idx: %lu, global_idx: %lu, shot_idx: %lu\n", state_idx,
    //        global_idx, shot_idx);
    return;
  }

  for (int i = 0; i < RAND_POSSIBLE_OUTCOME; i++) {
    uint64_t new_idx = global_idx + i * STATE_COUNTER;
    // printf("new_idx: %lu, global_idx: %lu, state_idx: %lu i: %d\n", new_idx,
    //        global_idx, state_idx, i);
    states[new_idx].a = DISTRIBUTION[i];
  }

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

  // task.params.push_back(
  //     {Parameter::ID, Parameter::RAND_OP, Parameter::Y_OP, Parameter::X_OP,
  //      Parameter::RAND_OP, Parameter::RAND_OP, Parameter::X_OP,
  //      Parameter::RAND_OP, Parameter::RAND_OP, Parameter::Z_OP});
  // task.params.push_back({Parameter::ID, Parameter::RAND_OP, Parameter::X_OP,
  //                        Parameter::Y_OP, Parameter::RAND_OP,
  //                        Parameter::X_OP, Parameter::RAND_OP,
  //                        Parameter::RAND_OP, Parameter::RAND_OP,
  //                        Parameter::Y_OP});
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
    task.params.push_back(
        {Parameter::ID, Parameter::RAND_OP, Parameter::X_OP, Parameter::RAND_OP,
         Parameter::X_OP, Parameter::RAND_OP, Parameter::X_OP,
         Parameter::RAND_OP, Parameter::RAND_OP, Parameter::ID});
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
  int device;
  cudaDeviceProp properties;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);

  printf("Max threads per block: %d\n", properties.maxThreadsPerBlock);
  printf("Max block dimensions: (%d, %d, %d)\n", properties.maxThreadsDim[0],
         properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", properties.maxGridSize[0],
         properties.maxGridSize[1], properties.maxGridSize[2]);

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

  printf("Maximum number of states: %lu\n", MAXIMUM);

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

  State *d_states = (State *)malloc(task.num_shots * num_blocks_per_shot *
                                    1024 * sizeof(State));

  for (int j = 0; j < task.num_shots * num_blocks_per_shot * 1024; j++) {
    d_states[j].a = 0.0;
  }

  status =
      cudaMemcpy((void *)states_ptr, (void *)d_states,
                 num_blocks_per_shot * 1024 * task.num_shots * sizeof(State),
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

    // printf("Start executing parameter %d\n", iter_param);
    // printf("STATE_COUNTER: %lu\n", STATE_COUNTER);

    bool all_rand_op = true;
    bool has_rand_op = false;

    std::vector<int> rand_op_idx;
    std::vector<int> non_rand_idx;

    for (int e = 0; e < task.num_shots; e++) {
      for (int p = 0; p < NUM_PARAMETERS; p++) {
        // printf("Shot %d, parameter %d: %d\n", e, p, params[e][p]);
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
      printf("Adding padding ID gates\n");
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
    }

    // for (int e = 0; e < task.num_shots; e++) {
    //   for (int p = 0; p < NUM_PARAMETERS; p++) {
    //     printf("Shot %d, parameter %d: %d\n", e, p, params[e][p]);
    //   }
    // }

    cudaDeviceSynchronize();

    if (all_rand_op) {
      // printf("Launching random kernel\n");
      run_random<<<task.num_shots * num_blocks_per_shot, 1024>>>(
          states_ptr, task.num_shots, num_blocks_per_shot, iter_param);

      cudaDeviceSynchronize();

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
      }

      
      STATE_COUNTER *= RAND_POSSIBLE_OUTCOME;
    } else {
      // printf("Launching kernel\n");
      run<<<task.num_shots * num_blocks_per_shot, 1024>>>(
          states_ptr, task.num_shots, num_blocks_per_shot, iter_param);

      cudaDeviceSynchronize();

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
      }
    }

    // printf("Kernel returns\n");

    cudaDeviceSynchronize();

    // printf("STATE_COUNTER: %lu\n", STATE_COUNTER);
    // printf("Finish executing parameter %d\n", iter_param);

    iter_param++;
    // printf("Increamenting iter_param to %d\n", iter_param);
  }

  printf("Start reducing\n");

  cudaDeviceSynchronize();

  cudaMemcpy((void *)d_states, (void *)states_ptr,
             sizeof(State) * num_blocks_per_shot * 1024 * task.num_shots,
             cudaMemcpyDeviceToHost);

  std::vector<std::unordered_map<float, uint64_t>> stats(task.num_shots);

  for (int e = 0; e < task.num_shots; e++) {
    for (int i = 0; i < MAXIMUM; i++) {
      // printf("HOST: Shot %d, state %d: %f\n", e, i, d_states[e][i].a);
      stats[e][d_states[e * num_blocks_per_shot * 1024 + i].a]++;
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
  delete[] d_states;
}
