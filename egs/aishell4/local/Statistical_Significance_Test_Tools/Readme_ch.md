# Statistical Significance Test Tools

## Overview

- significance_test.py: 此Python脚本提供了两种统计假设检验方法，用于比较两个相关实验结果之间的显著性差异：McNemar检验和匹配对检验。
- cer.py: 此 Python 脚本用于计算ground_truth和hypothesis之间的字符错误率（CER）。它专为处理包含序列的文本文件设计，计算每对相应句子的 CER。
- p_cal.sh: 此 Bash 脚本自动化了对两个实验进行字符错误率（CER）计算以及它们结果进行显著性检验的过程。它利用了提供的 Python 脚本进行CER计算（cer.py）和显著性检验（significance_test.py）

## 使用方法

### 依赖

确保你的环境中安装了以下依赖：

- Python 3.x
- NumPy
- SciPy
- jiwer

你可以使用以下命令安装依赖：

```bash
pip install numpy scipy jiwer
```

## 运行脚本

### p_cal.sh使用
```bash
./p_cal.sh <ground_truth_file> <cache_folder> <calculation_type> <exp1_result_file> <exp2_result_file>
```
- <ground_truth_file>: ground_truth文件的路径。
- <cache_folder>: 缓存文件夹的路径，用于存储中间结果。
- <calculation_type>: 要执行的显著性检验的类型（mc 表示 McNemar 检验，mp 表示匹配对检验）。
- <exp1_result_file>: 实验1结果文件的路径。
- <exp2_result_file>: 实验2结果文件的路径。

#### 示例：
- 确保脚本具有执行权限：
```bash
chmod +x p_cal.sh
```
- 运行脚本并提供所需的参数：
```bash
./p_cal.sh path/to/ground_truth.txt path/to/cache_folder mc path/to/exp1_results.json path/to/exp2_results.json
```
#### 输出
脚本将执行以下操作：

1. 对实验1运行 CER 计算，结果保存到 $cache_folder/cer_results_exp1.json。
2. 对实验2运行 CER 计算，结果保存到 $cache_folder/cer_results_exp2.json。
3. 运行指定类型的显著性检验，比较两个实验的结果。

=================================================================
### cer.py使用
在命令行中运行脚本，提供ground_truth和hypothesis的路径：
```bash
python cer.py path/to/ground_truth.txt path/to/hypothesis.txt --cer --force-cased --output-path cer_results.json
```
- 替换 path/to/ground_truth.txt 和 path/to/hypothesis.txt 为ground_truth和hypothesis的路径。

#### 参数说明:
- --cer: 计算 CER，如果提供此选项，默认为 False。
- --force-cased: 强制文本保持相同的大小写。
- --output-path: 保存 CER 结果的 JSON 文件路径，默认为 cer_results.json。

#### 输出
运行脚本后，将输出计算得到的 CER 结果，并将其保存到指定的 JSON 文件中。

=================================================================
### significance_test.py使用
在命令行中运行脚本，并传递两个实验结果文件的路径：

```bash
python significance_test.py path/to/exp1_results.json path/to/exp2_results.json --method mc
```

- 替换 path/to/exp1_results.json 和 path/to/exp2_results.json 分别为你的实验结果文件的路径。
- 使用 --method mc 表示选择McNemar检验，你也可以使用 --method mp 选择匹配对检验。


#### 参数说明:
- result_path1: exp1的结果文件路径（JSON格式）
- result_path2: exp2的结果文件路径（JSON格式）
- --method: 选择检验方法，可选值为 mc 或 mp，默认为 mp
#### 输出
运行脚本后，将输出所选择检验方法的概率值 P，用于评估两个实验结果之间的统计显著性差异。

## 了解更多
有关显著性检验相关知识，请参见：[显著性检验相关知识](https://mp.weixin.qq.com/s/i-z0Okeh76KBFNhdLh2Luw)。

