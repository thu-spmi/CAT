### Basic info

**This part is auto generated, add your details in Appendix**

* Model size/M: 70.59
* GPU info \[5\]
  * \[1\] NVIDIA GeForce GTX 1080 Ti
  * \[4\] NVIDIA GeForce GTX 1080

### Appendix

* Multilingual training with `JoinAP Nonlinear` mode on the data pooled of `de`, `fr`, `it` and `es` from CommonVoice 5.1.

* Crosslingual training with `JoinAP Nonlinear` mode on the data pooled of `pl`, `zh` from CommonVoice 5.1 and aishell-1.

### Multilingual WER

|language|w/o finetune| w/ [finetune](./Finetune/)|
|---|---|---|
|de|13.95|12.89|
|fr|24.61|20.39|
|it|24.21|21.18|
|es|15.01|13.23|

### Crosslingual WER

|language|w/o finetune| w/ [finetune](./Finetune)|
|---|---|---|
|pl|30.56|8.26 (10min)|
|zh|89.10|23.69 (1h)|


### Monitor figure
![monitor](./monitor.png)
