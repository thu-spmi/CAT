#!/bin/bash

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
