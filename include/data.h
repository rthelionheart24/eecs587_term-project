#ifndef DATA_H
#define DATA_H

#include <vector>   

typedef enum gate{
    H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP, MEASURE
} gate;

typedef struct complex{
    float real;
    float imag;
} complex;

typedef struct QCircuit{
    
    std::vector<complex> qubits;
    std::vector<complex> bits;
    std::vector<gate> gates;

    unsigned int num_qubits;
    unsigned int num_bits;
    unsigned int num_gates;


    QCircuit(unsigned int num_qubits, unsigned int num_bits, unsigned int num_gates){
        this->num_qubits = num_qubits;
        this->num_bits = num_bits;
        this->num_gates = num_gates;

        this->gates = std::vector<gate>(num_gates);
        this->qubits = std::vector<complex>(num_qubits);
        this->bits = std::vector<complex>(num_bits);
    };
} QCircuit;



#endif