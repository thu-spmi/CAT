# Significance Test

To see whether the difference between two experiments is significant, we need to conduct significance test and calculate the $p$ value. If we set the significance level $\alpha=0.05$ (typical values are 0.05, 0.01 and 0.001), then all the experiment pairs with $p$ value less than 0.05 are considered to be significantly different.

```bash
python utils/significance_test.py ${result_path1} ${result_path2} --method mp
```

`result_path1` and `result_path2` denote the metric values on all the test samples extracted from the results of the two experiments. `--method mp` denotes matched pair test and you can also set `--method mc`, which denotes McNemar test. Noting that the metric value can only be 0 or 1 in McNemar test.

### References

L. Gillick and S. J. Cox, “Some statistical issues in the comparison of speech recognition algorithms,” in International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 1989, pp.532–535.