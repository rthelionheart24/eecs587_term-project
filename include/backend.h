#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#include "data.h"


class BatchedGPUTask{
    public:
      BatchedGPUTask(QCircuit *qc, unsigned int num_shots);

      void runWrapper();

      ~BatchedGPUTask();

    private:
    /*
    * The circuit to be executed on the GPU
    */
    QCircuit *circuit;

      unsigned int num_shots;

      unsigned int qubits_start_idx, bits_start_idx, gates_start_idx;

};


#endif