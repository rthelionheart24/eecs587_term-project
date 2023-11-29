#!/bin/bash
#SBATCH --job-name=term-project
#SBATCH --nodes=1
#SBATCH--gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f23_class
#SBATCH --partition=gpu


module load cuda
module load gcc

# Change directory to the project root
cd /home/qifwang/eecs587/term-project/build
rm -rf *
cmake ..
make


cd /home/qifwang/eecs587/term-project/build

rm ../output.txt

./scheduler > ../output.txt