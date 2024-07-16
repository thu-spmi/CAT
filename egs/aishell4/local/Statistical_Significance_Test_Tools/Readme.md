# Statistical Significance Test Tools

## Overview

- significance_test.py: This Python script provides two statistical hypothesis testing methods for comparing the significance difference between two related experimental results: the McNemar test and the Matched-Pairs test.
- cer.py: This Python script is used to calculate the Character Error Rate (CER) between ground truth and hypothesis sequences. It is designed to handle text files containing sequences and computes the CER for each corresponding pair of sentences.
- p_cal.sh: This Bash script automates the process of calculating the CER for two experiments and conducting significance tests on their results. It leverages the provided Python scripts for CER calculation (cer.py) and significance testing (significance_test.py).

## Usage Instructions

### Dependencies
Ensure your environment has the following dependencies installed:

- Python 3.x
- NumPy
- SciPy
- jiwer

You can install the dependencies using the following command:

```bash
pip install numpy scipy jiwer
```

## Running the Scripts

### Using p_cal.sh

```bash
./p_cal.sh <ground_truth_file> <cache_folder> <calculation_type> <exp1_result_file> <exp2_result_file>
```

- <ground_truth_file>: Path to the ground truth file.
- <cache_folder>: Path to the cache folder for storing intermediate results.
- <calculation_type>: Type of significance test to perform (mc for McNemar test, mp for Matched-Pairs test).
- <exp1_result_file>: Path to the result file of experiment 1.
- <exp2_result_file>: Path to the result file of experiment 2.

#### Example:
Ensure the script has execution permissions:

```bash
chmod +x p_cal.sh
```

Run the script with the required parameters:

```bash
./p_cal.sh path/to/ground_truth.txt path/to/cache_folder mc path/to/exp1_results.json path/to/exp2_results.json
```

#### Output
The script will perform the following actions:

1. Run CER calculation for experiment 1 and save the results to $cache_folder/cer_results_exp1.json.
2. Run CER calculation for experiment 2 and save the results to $cache_folder/cer_results_exp2.json.
3. Run the specified type of significance test to compare the results of the two experiments.

=================================================================

### Using cer.py

Run the script from the command line, providing paths to the ground truth and hypothesis files:

```bash
python cer.py path/to/ground_truth.txt path/to/hypothesis.txt --cer --force-cased --output-path cer_results.json
```

- Replace path/to/ground_truth.txt and path/to/hypothesis.txt with the paths to your ground truth and hypothesis files.

#### Parameters:
- --cer: Calculate CER. If this option is provided, it defaults to False.
- --force-cased: Force the text to maintain the same casing.
- --output-path: Path to save the CER results in a JSON file, default is cer_results.json.

#### Output
After running the script, it will output the calculated CER results and save them to the specified JSON file.

=================================================================

### Using significance_test.py
Run the script from the command line, passing the paths to the two experimental result files:

```bash
python significance_test.py path/to/exp1_results.json path/to/exp2_results.json --method mc
```

- Replace path/to/exp1_results.json and path/to/exp2_results.json with the paths to your experimental result files.
- Use --method mc to select the McNemar test, or use --method mp to select the Matched-Pairs test.

#### Parameters:
- result_path1: Path to the result file of experiment 1 (in JSON format).
- result_path2: Path to the result file of experiment 2 (in JSON format).
- --method: Select the testing method, with options mc or mp. Default is mp.

#### Output
After running the script, it will output the p-value of the chosen significance test method, which is used to evaluate the statistical significance difference between the two experimental results.

## Learn More
For more information on significance testing, please refer to: Significance Testing Knowledge.