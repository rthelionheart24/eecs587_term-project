#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <curand_kernel.h>

#define THREADS_PER_BLOCK 1024



enum struct Parameter { ADD, SUB, MULT, DIV, RAND };

typedef struct QData {
  float a;
  float b;
} QData;

typedef struct Task {
  std::vector<Parameter> params;
  QData data;
  uint64_t num_shots;
} Task;


__managed__ uint64_t statesCounter = -1;

// __managed__ Parameter* params;

__device__ void printParam(Parameter *param, uint64_t paramsCounter) {

  switch (*param) {

  case Parameter::ADD:
    printf("Parameter: ADD\n");
    break;
  case Parameter::SUB:
    printf("Parameter: SUB\n");
    break;
  case Parameter::MULT:
    printf("Parameter: MULT\n");
    break;
  case Parameter::DIV:
    printf("Parameter: DIV\n");
    break;
  case Parameter::RAND:
    printf("Parameter: RAND\n");
    break;
  }
}

__device__ void printData(QData *data, uint64_t numBlocksParams) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t data_idx = idx - numBlocksParams * THREADS_PER_BLOCK;

  printf("Data at position %lu: (%f, %f)\n", idx, data[data_idx].a,
         data[data_idx].b);
}

__global__ void run(Parameter *params, QData *data, uint64_t num_shots,
                    uint64_t num_params, uint64_t numBlocksParams,
                    uint64_t numBlocksData, uint64_t data_idx,
                    uint64_t params_idx,
                    uint64_t paramsCounter) {

  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < data_idx) {
    return;
  }

  if (global_idx > data_idx + statesCounter) {
    return;
  }

  printParam(&params[paramsCounter], paramsCounter);
  printData(data, numBlocksParams);

  // Transformation
  if (params[paramsCounter] == Parameter::ADD) {
    data[data_idx].a += data[data_idx].b;
  } else if (params[paramsCounter] == Parameter::SUB) {
    data[data_idx].a -= data[data_idx].b;
  } else if (params[paramsCounter] == Parameter::MULT) {
    data[data_idx].a *= data[data_idx].b;
  } else if (params[paramsCounter] == Parameter::DIV) {
    // Ensure that data[data_idx].b is not zero to avoid division by zero
    if (data[data_idx].b != 0) {
      data[data_idx].a /= data[data_idx].b;
    }
  } else if (params[paramsCounter] == Parameter::RAND) {
    // Note: std::rand() is generally not recommended in CUDA kernels
    // Consider using curand for random number generation in CUDA
  


  }



  return;
}

int main(int argc, char **argv) {

  QData d{1.0, 3.0};
  std::vector<Parameter> p{Parameter::ADD, Parameter::SUB, Parameter::MULT,
                           Parameter::DIV, Parameter::RAND};

  Task *task;
  task = (Task *)malloc(sizeof(Task));
  task->params = p;
  task->data = d;

  uint64_t params_idx = 0, data_idx;

  Parameter *params_ptr;
  QData *data_ptr;

  uint64_t numBlocksParams = 1, numBlocksData;

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

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);  // Setting a seed for reproducibility


  dim3 gridDim;
  gridDim.x = 0;

  cudaError_t status;

  // printf("Start allocating for params at location: %f\n", d_params);
  status =
      cudaMalloc((void **)&params_ptr, sizeof(Parameter) * task->params.size());
  if (status != cudaSuccess) {
    printf("Error allocating memory for params - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for params\n");
  }

  gridDim.x += task->params.size() / THREADS_PER_BLOCK;
  if (task->params.size() % THREADS_PER_BLOCK != 0) {
    gridDim.x++;
  }

  //// dataOriginal = paramsOriginal + gridDim.x * THREADS_PER_BLOCK;
  //// printf("Start allocating memory for data at location: %f\n",
  /// dataOriginal);

  status = cudaMalloc((void **)&data_ptr, sizeof(QData));
  if (status != cudaSuccess) {
    printf("Error allocating memory for data - %s\n",
           cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Allocated memory for data\n");
  }

  gridDim.x++;

  printf("Grid dim: %d\n", gridDim.x);
  printf("Block dim: %d\n", THREADS_PER_BLOCK);

  data_idx = numBlocksParams * THREADS_PER_BLOCK;

  // =================================================================================================================
  // Copy data to device

  status = cudaMemcpy((void *)params_ptr, task->params.data(),
                      sizeof(Parameter) * task->params.size(),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    printf("Error copying params to device - %s\n", cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied params of size %zu to device\n", task->params.size());
  }

  status = cudaMemcpy((void *)data_ptr, &(task->data), sizeof(QData),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    printf("Error copying data to device - %s\n", cudaGetErrorString(status));
    exit(EXIT_FAILURE);
  } else {
    printf("Copied data of length to device\n");
  }

  // =================================================================================================================
  // Running task

  printf("Running task\n");

  statesCounter = 0;
  uint64_t paramsCounter = 0;

  for (int i = 0; i < task->params.size(); i++) {
    if (task->params[i] != Parameter::RAND) {
          // Run the simulation kernel
    run<<<gridDim, THREADS_PER_BLOCK>>>(params_ptr, data_ptr, task->num_shots,
                                        task->params.size(), numBlocksParams,
                                        numBlocksData, data_idx, params_idx,
                                        paramsCounter);
    }
    else{
      statesCounter++;

      


    }



    cudaDeviceSynchronize();
  }

  // =================================================================================================================
  // Benchmarking

  cudaFree(params_ptr);
  cudaFree(data_ptr);
  free(task);
}
