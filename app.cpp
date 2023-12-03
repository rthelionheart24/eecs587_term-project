#include <cstdlib>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>


#include "include/backend.h"
#include "include/data.h"

using namespace std;

int main(int argc, char* argv[]) {


    printf("App started\n");

    std::string stars(50, '*');
    std::string dashes(50, '-');


    printf("%s\n", stars.c_str());

 

    // ? Construct a circuit and feed it to the batch-scheduler

    

    printf("%s\n", stars.c_str());

    printf("App finished\n");
    return 0;

}