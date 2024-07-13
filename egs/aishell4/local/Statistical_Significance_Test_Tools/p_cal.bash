#!/bin/bash

# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Xiangzhu Kong (kongxiangzhu99@gmail.com)
#
# Description:
#   This script calculates the Character Error Rate (CER) for two sets of experimental results and performs a significance test to compare them.
#   The script takes in a ground truth file, a cache folder for storing intermediate results, the type of calculation for significance testing, 
#   and the result files from two experiments. It then runs the CER calculations for each experiment and saves the results in the specified cache folder.
#   Finally, it performs a significance test comparing the CER results of the two experiments.

# Check if all required arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <ground_truth_file> <cache_folder> <calculation_type> <exp1_result_file> <exp2_result_file>"
    exit 1
fi

# Set script arguments
ground_truth=$1
cache_folder=$2
calculation_type=$3
exp1_result=$4
exp2_result=$5


# Run CER for experiment 1
python local/cer_cal/cer.py "$ground_truth" "$exp1_result" --cer --out "$cache_folder/cer_results_exp1.json"
echo "CER results for Experiment 1 saved to $cache_folder/cer_results_exp1.json"

# Run CER for experiment 2
python local/cer_cal/cer.py "$ground_truth" "$exp2_result" --cer --out "$cache_folder/cer_results_exp2.json"
echo "CER results for Experiment 2 saved to $cache_folder/cer_results_exp2.json"

# Run significance test
python local/cer_cal/significance_test.py "$cache_folder/cer_results_exp1.json" "$cache_folder/cer_results_exp2.json" --me "$calculation_type"
