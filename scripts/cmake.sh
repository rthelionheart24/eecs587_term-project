#!/bin/bash

module load cuda
module load gcc

# Change directory to the project root
cd /home/qifwang/eecs587/term-project/build
rm -rf *
cmake ..
make
