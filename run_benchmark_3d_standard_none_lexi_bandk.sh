#!/bin/bash


cd "C:\Users\Bolt\Documents\GitRepositories\HiParTI\cmake-default-release\benchmark\tensor"
export OMP_NUM_THREADS=4
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 8 -l 8 -d -1 -e 0
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 8 -l 8 -d -1 -e 1
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 8 -l 8 -d -1 -e 4