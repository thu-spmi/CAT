## Join Acoustics and Phonology (JoinAP) for Multi/Cross-lingual ASR 

---

This page introduces how to conduct **Multi/Cross-lingual ASR** study with the recently proposed **JoinAP** method. We recommend to read the references at the bottom of this page for theoretical understanding and `CAT/scripts/ctc-crf/{model.py,mc_lingual.py}` for implementation details. 

In this page, we will first introduce how to conduct  **Multi-lingual** experiments, with **Flat-phone**, **JoinAP-Linear** and **JoinAP-Nonlinear**, respectively. Finally, **Cross-lingual** experiments will be described.

### What we will cover

---

* [Flat-phone](#flat-phone)

* [JoinAP-Linear](#joinap-linear)

* [JoinAP-Nonlinear](#joinap-nonlinear)

* [Cross-lingual](#cross-lingual)

---

<a name="flat-phone">**Flat-phone**</a>

- #### training 

  - **Flat-phone** based multilingual training is the same as monolingual training.

- #### testing with finetune

  - Testing with finetuning consists of two stages. The first stage is finetuning, and the second stage is testing with the finetuned model. The two stages for **Flat-phone** system is shown as follows:

    ``` shell
    dir=exp/mc_flatphone
    finetune_dir=exp/mc_flatphone_finetune
    
    # finetune
    if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
        unset CUDA_VISIBLE_DEVICES
        if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
            echo ""
            tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
        elif [ $NODE == 0 ]; then
            echo ""
            echo "'$dir/scripts.tar.gz' already exists."
            echo "If you want to update it, please manually rm it then re-run this script."
        fi
    		for x in ${lang[@]}; do	
        		CUDA_VISIBLE_DEVICES=0,1,2   					          			\
        		python3 ctc-crf/train.py --seed=0               			\
            		--world-size 1 --rank $NODE                 			\
            		--batch_size=128                            			\
            		--resume=$dir/ckpt/bestckpt.pt              			\
            		--den-lm=data/den_meta_${x}/den_lm.fst        		\
            		--mc-conf=./conf/mc_flatphone_finetune_${x}.json 	\
            		--trset=data/pickle/train_${x}.pickle         		\
            		--devset=data/pickle/dev_${x}.pickle          		\
            		--dir=$finetune_dir                         			\
            		--config=$dir/config.json                   			\
            		--data=data/train_${x}_sp || exit 1;
        done
    fi
    
    # testing
    if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
        for x in ${lang[@]}; do
            scp=data/all_ark/test_${x}.scp
            ark_dir=$finetune_dir/decode_${x}_test_bd_tgpr/logits
            mkdir -p $ark_dir
            CUDA_VISIBLE_DEVICES=0,1,2,3        							  \
            python3 ctc-crf/calculate_logits.py                 \
                --mc-conf=./conf/mc_flatphone_eval_${x}.json    \
                --resume=$finetune_dir/ckpt/infer.pt            \
                --dist-url="tcp://127.0.0.1:13986"              \
                --config=$finetune_dir/config.json              \
                --nj=$nj --input_scp=$scp                       \
                --output_dir=$ark_dir                           \
                || exit 1
    
            ctc-crf/decode.sh  --stage 1 --cmd "$decode_cmd" --nj $nj --acwt 1.0 					\
            		data/lang_phn_${x}_test_bd_tgpr data/test_${x} data/all_ark/test_${x}.ark \
                $finetune_dir/decode_${x}_test_bd_tgpr || exit 1
        done
    fi
    ```

    Multi/Cross-lingual finetuning is implemented with `ctc/crf/train.py` by adding additional arguments. For `Flat-phone` finetuning, `--den-lm` and `--mc-conf` are used to feed two input files:
    - `--den-lm` specifies the file used to construct the `denominator graph` for the finetuned language;
    - `--mc-conf` specifies the configuration file used to adjust the model for finetuning. 
    We take `de` as the language for finetuning and the corresponding file of `conf/mc_flatphone_finetune_de.json` is shown as below:

    ```json
    {
        "src_token": "./data/lang_phn/tokens.txt", 
        "des_token": "./data/lang_phn_de/tokens.txt",
        "P": null,
      	"lr": 1e-5,
        "hdim": 2048,
        "odim": 43,
        "mode": "flat_phone",
        "usg": "finetune"
    }
    ```

     Each field in the configuration json file is as follows:
      - `src_token` and `des_token`: used to build the mapping relationship of output units between the original model and the target model; 
      - `P`: used to specify the phonological vector file for **JoinAP** method;
      - `lr`: used to specify the initial learning rate for finetuning; 
      - `hdim` and `odim` depends on the target language (to be detailed);
      -  `mode`: specify the model type, must be one of `["flat_phone", "joinap_linear", "joinap_nonlinear"]`;
      -  `usg`: specify the behavior for the model, must be one of `["fientune", "zero-shot-eval", "few-shot-eval", "multi-eval", "multi-finetune-eval"]`. 

        - `"finetune"` is used for finetuning for both multi and cross lingual exps. 
        - `"multi-eval"` and `"multi-finetune-eval"` are used for testing multi-lingual experiments;
        - `"zero-shot-eval"` and `"few-shot-eval"` are used for testing cross-lingual experiments;
      
      For the testing without finetuning of `Flat-phone` model, you should specify `"mode"` as `"flat-phone"` and `"usg"`   as `"zero-shot-eval"` . Once `"mode"` is set to `"flat_phone"`, the value of `"P"` will not take effect. `"lr"` will only take effect when `"usg"` is set to `"finetune"`. 
      
    Testing of the target languages can be conducted once finetuning finished, which is shown in `$stage=9` in the above code. Multi/Cross-lingual evaluation is implemented with `ctc/calculate_logits.py` by adding some additional arguments. Care should be taken to set `--resume`, `--config` and the parameters for `ctc-crf/decode.sh`. `conf/mc_flatphone_finetune_eval_de.json` should be the same with that of `conf/mc_flatphone_finetune_de.json`,  except that `"usg"` should be set to `"few-shot-eval"` instead.

- #### testing without finetune

  Testing without finetune only involves the final evaluation stage, thus the finetune stage (stage=8 for the most time) should be commented if existed.

```shell
dir=exp/mc_flatphone
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    for x in ${lang[@]}; do
        scp=data_${x}/test_data/test.scp
        ark_dir=$dir/decode_${x}_test_bd_tgpr/logits
        mkdir -p $ark_dir
        CUDA_VISIBLE_DEVICES=0                              \
        python3 ctc-crf/calculate_logits.py                 \
            --mc-conf=./conf/mc_flatphone_${x}_eval.json    \
            --resume=$dir/ckpt/infer.pt                     \
            --dist-url="tcp://127.0.0.1:13986"              \
            --config=$dir/config.json                       \
            --nj=$nj --input_scp=$scp                       \
            --output_dir=$ark_dir                           \
            || exit 1

        ctc-crf/decode.sh  --stage 1 --cmd "$decode_cmd" --nj $nj --acwt 1.0 						\
        		data/lang_phn_${x}_test_bd_tgpr data/test_$x data/test_${x}_data/test.ark 	\
        		$dir/decode_${lang}_test_bd_tgpr || exit 1
    done
fi

```

​Here `--den-lm` is very important and should be set appropriately based on the language for finetuning. You should also be careful to set `--resume` and `--dir`. The config file for `--mc-conf` is the same to that in the `Flat-phone` model testing without finetuning, except that `"usg"` should be set as `"finetune"`. 

<a name="joinap-linear">**JoinAP-Linear**</a>

- #### training

  **JoinAP-Linear** multilingual training is enabled with `ctc-crf/train.py` by add additional  parameter of `--mc-train-pv` , which specifies the `phonological vector` (abbreviated as `pv`) used for multi-lingual training. A typical configuration is shown as below:

  ```shell
  
  dir="exp/mc_linear/"
  DATAPATH=$PARENTDIR/data/
  if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
      unset CUDA_VISIBLE_DEVICES
      if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
          echo ""
          tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
      elif [ $NODE == 0 ]; then
          echo ""
          echo "'$dir/scripts.tar.gz' already exists."
          echo "If you want to update it, please manually rm it then re-run this script."
      fi
      
      CUDA_VISIBLE_DEVICES=0,1,2                      \
      python3 ctc-crf/train.py --seed=0               \
          --world-size 1 --rank $NODE                 \
          --mc-train-pv=./saved_pv/mul.npy            \
          --batch_size=128                            \
          --dir=$dir                                  \
          --config=$dir/config.json                   \
          --trset=data/pickle/train.pickle            \
          --devset=data/pickle/dev.pickle             \
          --data=$DATAPATH                            \
          || exit 1
  fi
  ```

  

- #### testing with finetune

  Similar to the aforementioned **Flat-phone** system, testing with finetuning for **JoinAP-Linear** also consists of two stages, which is shown as below:

  ``` shell
  dir="exp/mc_linear"
  finetune_dir="exp/mc_linear_finetune/"
  if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
      # finetune
      unset CUDA_VISIBLE_DEVICES
      if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
          echo ""
          tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
      elif [ $NODE == 0 ]; then
          echo ""
          echo "'$dir/scripts.tar.gz' already exists."
          echo "If you want to update it, please manually rm it then re-run this script."
      fi
  		for x in ${lang[@]}; do
      		CUDA_VISIBLE_DEVICES=0,1,2                      	\
      		python3 ctc-crf/train.py --seed=0               	\
          		--world-size 1 --rank $NODE                 	\
          		--batch_size=128                            	\
          		--mc-train-pv=./saved_pv/mul.npy            	\
          		--resume=$dir/ckpt/bestckpt.pt              	\
          		--den-lm=data/den_meta_${x}/den_lm.fst        \
          		--mc-conf=./conf/mc_linear_finetune_${x}.json \
          		--trset=data/pickle/train_${x}.pickle         \
          		--devset=data/pickle/dev_${x}.pickle          \
          		--dir=$finetune_dir                         	\
          		--config=$dir/config.json                   	\
          		--data=data/train_${x}_sp || exit 1;
        done
  fi
  
  if [ $NODE -ne 0 ]; then
      exit 0
  fi
  nj=32
  
  if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
      for x in ${lang[@]}; do
          scp=data/all_ark/test_${x}.scp
          ark_dir=$finetune_dir/decode_${x}_test_bd_tgpr/logits
          mkdir -p $ark_dir
          CUDA_VISIBLE_DEVICES=1,2            											\
          python3 ctc-crf/calculate_logits.py    			             	\
              --mc-conf=./conf/mc_linear_finetune_eval_${x}.json    \
              --mc-train-pv=./saved_pv/${x}.npy            					\
              --resume=$finetune_dir/ckpt/infer.pt                  \
              --dist-url="tcp://127.0.0.1:13986"              			\
              --config=$finetune_dir/config.json                    \
              --nj=$nj --input_scp=$scp                       			\
              --output_dir=$ark_dir                           			\
              || exit 1
  
          ctc-crf/decode.sh  --stage 1 --cmd "$decode_cmd" --nj $nj --acwt 1.0							\
          		data/lang_phn_${x}_test_bd_tgpr data/test_${x} data/all_ark/test_${x}.ark			\
              $finetune_dir/decode_${x}_test_bd_tgpr || exit 1
      done
  fi
  ```

  **JoinAP-Linear** finetuning is enabled by passing additional parameters to `ctc-crf/train.py`, including both `--mc-conf` and `--mc-train-pv`. File of `--mc-train-pv` should be the same as that of model training stage, since it is used to resume from the pretrained model at the stage of finetuning. A typical `conf/mc_linear_finetune_de.json` used for finetuning is shown as below:

  ```json
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_de/tokens.txt",
      "P": "./saved_pv/de.npy",
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 43,
      "mode": "joinap_linear",
      "usg": "finetune"
  }
  ```

  Testing after finetuning **JoinAP-Linear** model is enabled by passing additional parameters to `ctc-crf/calculate_logits.py`, including `--mc-train-pv` and `--mc-conf`. In this case, `--mc-train-pv` should be set to the path of taget language's phonological vector file. As for the `conf/mc_linear_finetune_eval_de.json`, which should be the same as that of `conf/mc_linear_finetune_de.json` (shown as above), except `"usg"` should be set as `"multi-finetune-eval"`.

- #### testing without finetunine 

  Testing without finetuning for **JoinAP-Linear** model will evaluate the trained model directly on the target language. The evaluation process is the same as that of stage-9 in the above **JoinAP-Linear** testing with finetuning, but the file `conf/mc_linear_eval.json` to `--mc-conf` should be set as below:

  ```json
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_de/tokens.txt",
      "P": "./saved_pv/de.npy",
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 43,
      "mode": "joinap_linear",
      "usg": "multi-eval"
  }
  ```

  

<a name="joinap-nonlinear">**JoinAP-Nonlinear**</a>

​	Model training, evaluation w/ or w/o finetuning for **JoinAP-Nonlinear** with **JoinAP-Linear** only differ in the configuration `json` file. 	You can simply set `"mode"` as `"joinap_nonlinear"` and reuse the above code of **JoinAP-Linear** for the **JoinAP-Nonlinear** exps. 

By far, we have went through all the models for **Multilingual** exps. Next, we will compare the differences of model evaluation between **Multi-lingual** againest **Cross-lingual** exps.

<a name="cross-lingual">**Cross-lingual**</a>

​	 We take `aishell` as the target language data for the **Cross-lingual** exps.

- **few-shot cross-lingual**

  For the **Flat-phone few-shot cross-lingual eval** , please refer to the above code of **testing with finetuning for multi-lingual Flat-phone**. The `conf/mc_flatphone_cross_zh.json` is shown as below:   

  ```json
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_zh/tokens.txt",
      "P": null,
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 99,
      "mode": "flat_phone",
      "usg": "finetune"
  }
  ```

  For the **JoinAP few-shot cross-lingual eval**, please refer to the above code of **testing with finetuning for multi-lingual JoinAP-Linear/Nonlinear**. The `conf/mc_cross_finetune_zh.json` and `conf/mc_cross_finetune_eval_zh.json` is shown as below:

  ```json
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_zh/tokens.txt",
      "P": "./saved_pv/zh.npy",
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 99,
      "mode": "joinap_linear",
      "usg": "finetune"
  }
  
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_zh/tokens.txt",
      "P": "./saved_pv/zh.npy",
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 99,
      "mode": "joinap_linear",
      "usg": "few-shot-eval"
  }
  ```

  

- **zero-shot cross-lingual**

  For the **Flat-phone zero-shot cross-lingual eval** , please refer to the above code of **testing without finetuning for multi-lingual Flat-phone**. The `conf/mc_flatphone_cross_zh.json` is shown as below:  

  ```json
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_zh/tokens.txt",
      "P": null,
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 99,
      "mode": "flat_phone",
      "usg": "zero-shot-eval"
  }
  ```

  For the **JoinAP zero-shot cross-lingual eval** , please refer to the above code of **testing without finetuning for multi-lingual JoinAP-Linear/Nonlinear**. The  `conf/mc_cross_eval_zh.json` is shown as below:

  ```json
  {
      "src_token": "./data/lang_phn/tokens.txt", 
      "des_token": "./data/lang_phn_zh/tokens.txt",
      "P": "./saved_pv/zh.npy",
    	"lr": 1e-5,
      "hdim": 2048,
      "odim": 99,
      "mode": "joinap_nonlinear",
      "usg": "zero-shot-eval"
  }
  ```

If you have encountered some problems in the **Multi/Cross-lingual** experiments, please feel free to file an issue.

### References

- Chengrui Zhu, Keyu An, Huahuan Zheng and Zhijian Ou, "Multilingual and crosslingual speech recognition using phonological-vector based phone embeddings", IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2021. [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)