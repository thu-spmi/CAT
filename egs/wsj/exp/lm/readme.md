
train command:

```bash
utils/pipeline/ngram.sh exp/lm -o 4 --arpa --prune 2 2 5 5 --output exp/lm/4gram.arpa --sto 4
```

property:

- prune: --prune 2 2 5 5
- type:  probing
- size:  63MB

perplexity:

```
data: test_dev93   test_eval92
ppl:   252.03    |  211.04  |
```
