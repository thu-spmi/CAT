
train command:

```bash
utils/pipeline/ngram.sh exp/lm-ngram-word -o 3
```

property:

- prune: 
- type:  probing
- size:  25MB

perplexity:

```
data: data/local-lm/libri-part.dev
ppl:   436.06  |
```
