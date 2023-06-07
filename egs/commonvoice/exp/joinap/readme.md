# Commonvoice + Join-AP experiments

Please first have a look at the [join-ap document](../../../../docs/joinap_tutorial_ch.md).

## Workflow

1. Download commonvoice data from https://commonvoice.mozilla.org/zh-CN/datasets

    In following experiments, we use Indonesian and Russian data (v11.0) as examples.

2. Prepare data

    ```bash
    # indonesian
    bash local/data.sh /path/to/cv-corpus-id -lang id
    # russian
    bash local/data.sh /path/to/cv-corpus-ru -lang ru
    ```

3. Prepare lexicons based on IPA.

    ```bash
    # get usage info.
    bash local/prep_ipa_lexicon.sh -h
    # prepare indonesian lexicon
    # bash local/prep_ipa_lexicon.sh -lang id -g2p_model /path/to/id_g2p.fst
    ```

4. Train NN models

    ```bash
    # indonesian monolingual
    ## flatphone
    python utils/pipeline/asr.py exp/joinap/mono-indonesia-flat
    ## join-ap linear
    python utils/pipeline/asr.py exp/joinap/mono-indonesia-L
    ## join-ap non-linear
    python utils/pipeline/asr.py exp/joinap/mono-indonesia-NL

    # indonesian + russian multilingual
    ## for short, we only give an example of join-ap linear.
    python utils/pipeline/asr.py exp/joinap/mul-ru+id-L
    ```

5. Construct decoding graph.

    ```bash
    # construct decoding graph
    ## indonesian
    ## train decoding lm
    bash utils/pipeline/ngram.sh exp/joinap/decode-lm-indonesia -o 3 --arpa
    ## obtain indonesian decoding graph for monolingual model
    bash utils/tool/build_decoding_graph.sh -c \
        exp/joinap/mono-indonesia-L/tokenizer.tknz \
        exp/joinap/decode-lm-indonesia/{tokenizer.tknz,3gram.arpa} \
        exp/joinap/mono-indonesia-L/graph

    ## obtain indonesian decoding graph for multilingual model
    bash utils/tool/build_decoding_graph.sh -c \
        exp/joinap/mul-ru+id-L/tokenizer.tknz \
        exp/joinap/decode-lm-indonesia/{tokenizer.tknz,3gram.arpa} \
        exp/joinap/mul-ru+id-L/graph-id

    ## russian ...
    ```

6. Decode with FST graph

    ```bash
    # monolingual
    bash ../TEMPLATE/local/eval_fst_decode.sh \
        exp/joinap/mono-indonesia-L/{,graph} \
        --data id-{dev,test}

    # multilingual
    bash ../TEMPLATE/local/eval_fst_decode.sh \
        exp/joinap/mul-ru+id-L/{,graph-id} \
        --data id-{dev,test}

    bash ../TEMPLATE/local/eval_fst_decode.sh \
        exp/joinap/mul-ru+id-L/{,graph-ru} \
        --data ru-{dev,test}
    ```

## Finetune a pretrained multilingual model on monolingual data

1. Unpack monolingual params from a multilingual pretrained model

    ```bash
    python exp/joinap/unpack_mulingual_param.py \
        exp/joinap/{mul-ru+id-L,mono-indonesia-L}/tokenizer.tknz \
        /path/to/multilingual/checkpoint \
        /path/to/output/checkpoint
        --mode joinap-linear
    ```

    Here, `--mode` can be one of `flatphone, joinap-linear, joinap-nonlinear`, which shoul be matched with your model configuration.

    Configure `train:option` to specify the initializing model. You can set either `init_model` or `resume`.

2. Train the monolingual model with initialized parameters. Please refer to [training workflow](#workflow).


