
train command:

```bash
utils/pipeline/ngram.sh exp/lm/lm-v6 -o 3
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

rescore rnnt-v19 a=0.28 b=-0.5 beamwidth=16
dev     %SER 31.75 | %CER 4.25 [ 8729 / 205341, 123 ins, 635 del, 7971 sub ]
test    %SER 32.78 | %CER 4.47 [ 4688 / 104765, 45 ins, 404 del, 4239 sub ]
```
