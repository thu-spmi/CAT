
train command:

```bash
utils/pipeline/ngram.sh exp/lm-ngram-word -o 3
```

property:

- prune: 
- type:  probing
- size:  17MB

perplexity:

```
Test file: data/local-lm/ptb.valid.txt -> ppl: 252.96
Test file: data/local-lm/ptb.test.txt -> ppl: 268.65
```
