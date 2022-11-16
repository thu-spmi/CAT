#!/bin/bash
# author: Huahuan Zheng (maxwellzh@outlook.com)
set -e -u

# use any other gpu ids if you want
export CUDA_VISIBLE_DEVICES=9

mkdir -p .done
# audio data prepare (yesno)
[ ! -f local/.audio.done ] && {
    bash local/data.sh
    touch local/.audio.done
}

[ ! -f .done/rnnt ] && {
    echo "> RNN-T testing..."
    python utils/pipeline/asr.py exp/asr-rnnt --silent >/dev/null || {
        echo "RNN-T test failed."
        exit 1
    }
    touch .done/rnnt
}

[ ! -f .done/rnnt_cuside ] && {
    echo "> RNN-T CUSIDE testing..."
    python utils/pipeline/asr.py exp/asr-rnnt-cuside --silent >/dev/null || {
        echo "RNN-T CUSIDE test failed."
        exit 1
    }
    touch .done/rnnt_cuside
}

[ ! -f .done/ctc ] && {
    echo "> CTC testing..."
    python utils/pipeline/asr.py exp/asr-ctc --silent >/dev/null || {
        echo "CTC test failed."
        exit 1
    }
    touch .done/ctc
}

# corpus data prepare (ptb)
[ ! -f local/.corpus.done ] && {
    bash local/lm_data.sh
    touch local/.corpus.done
}

[ ! -f .done/nnlm ] && {
    echo "> NN LM testing..."
    python utils/pipeline/lm.py exp/lm-nn --silent >/dev/null || {
        echo "NN LM test failed."
        exit 1
    }
    touch .done/nnlm
}

[ ! -f .done/ngram ] && {
    echo "> N-gram LM testing..."
    bash utils/pipeline/ngram.sh exp/lm-ngram-word -o 3 >/dev/null || {
        echo "N-gram LM test failed."
        exit 1
    }
    touch .done/ngram
}

rm -rf .done

echo "$0 done."
exit 0
