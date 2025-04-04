#!/bin/bash

#./run_benchmark_3d_density_none.sh
#./run_benchmark_3d_density_lexi.sh
#./run_benchmark_3d_density_bandk.sh

cd "C:\Users\Bolt\Documents\GitRepositories\HiParTI\cmake-default-release\benchmark\tensor"


#./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 32 -l 1 -d -1 -e 4
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 16 -l 1 -d -1 -e 4
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 8 -l 1 -d -1 -e 4
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\standard_tensors\nell-2.tns" -o test.tns -t 4 -l 1 -d -1 -e 4