
train command:

```bash
utils/pipeline/ngram.sh exp/lm/lm-v2-word-3gram -o 3
```

property:

- prune: 
- type:  probing
- size:  26MB

perplexity:

```
using jieba default dict produces better results:
Test file: dev.tmp -> ppl: 788.34
Test file: test.tmp -> ppl: 840.97

with bigcidian dict:
ppl ~1000
```
