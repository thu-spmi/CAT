# Crosslingual Fine-tuning Whistle model
Over the CV-Lang10 dataset, we obtain the phoneme-based supervised pre-trained model [Whistle phoneme S](../../../cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md), which can be further fine-tuned with either phoneme labels or subword labels. All the experimental results presented below indicate the word error rate (WER).


## Fine-tuning with phoneme or subword labels over German and Polish dataset
German and Polish are unseen languages for crosslingual ASR. The training data of Commonvoice from an unseen language is divided into two scales (20 hours and 130 hours) to simulate different resource scenarios, while the test and validation data remain unchanged. The experiments are also the baseline of our LLM-P2G.

## Results
### WER of Whistle phoneme FT (Baseline)

| FT language | 20 hours | 130 hours |
| ------ | ------ | ------ |
| | w LM | w LM |
| Polish  | [16.27](../crosslingual/pl/Whistle_ft_phoneme_20h/readme.md)  | [4.30](../crosslingual/pl/Whistle_ft_phoneme_130h/readme.md) |
| German  | [30.71](../crosslingual/de/Whistle_ft_phoneme_20h/readme.md) | [15.73](../crosslingual/de/Whistle_ft_phoneme_130h/readme.md) |

### WER of Whistle subword FT (Baseline)

| FT language | 20 hours | | 130 hours | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| Polish | [17.59](../crosslingual/pl/Whistle_ft_subword_20h/readme.md) | [13.84](../crosslingual/pl/Whistle_ft_subword_20h/readme.md) | [5.84](../crosslingual/pl/Whistle_ft_subword_130h/readme.md) | [3.82](../crosslingual/pl/Whistle_ft_subword_130h/readme.md) |
| German | [27.78](../crosslingual/de/Whistle_ft_subword_20h/readme.md) | [28.04](../crosslingual/de/Whistle_ft_subword_20h/readme.md) | [14.09](../crosslingual/de/Whistle_ft_subword_130h/readme.md) | [14.01](../crosslingual/de/Whistle_ft_subword_130h/readme.md) |


