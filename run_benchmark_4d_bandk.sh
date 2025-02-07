#!/bin/bash


cd "C:\Users\Bolt\Documents\GitRepositories\HiParTI\cmake-default-release\benchmark\tensor"

./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\new_tensors\test_tensor_script_4d_128_32_32_32_C.tns" -o test.tns -t 32 -l 8 -d -1 -e 4
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\new_tensors\test_tensor_script_4d_128_128_128_128_C.tns" -o test.tns -t 32 -l 8 -d -1 -e 4
./mttkrp_hicoo_reorder.exe -i "C:\Users\Bolt\Documents\GitRepositories\HiParTI\data\tensors\new_tensors\test_tensor_script_5d_32_32_32_32_32_A.tns" -o test.tns -t 32 -l 8 -d -1 -e 4