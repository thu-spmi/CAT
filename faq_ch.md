

# CAT 使用常见问题汇总（中文版）



1. **训练过程中出现nan的原因？**

      一个可能的原因是神经网络输出单元数目与词典中的单元数目不匹配，此时应当重新确定输出单元数目（见2）。

 

2. **如何确定输出单元数目？**

     在数据集上完成数据准备步骤后，在data目录下会生成lang_*文件夹，其中会有一个token.txt文件，其中除了eps token和#开始的token外，其余token的数目总和即为神经网络对应的输出单元数目。

 

3. **验证集loss为负数的原因？**

      由于验证集语料未纳入分母图语言模型训练，验证集上的p(l)计算可能出现问题，此时会出现loss为负数的情况。

 

4. **写文件时出现Error closing TableWriter [in destructor]**

      一般是因为磁盘满了，应当变更存储目录。

 

5. **加载模型时出现missing key(s)… unexpted key(s)**

    加载的模型与训练时定义的模型不同。如果出现size mismatch错误，则是隐藏单元数目等超参数定义与训练时定义的模型不同。一般而言，加载时的模型应当与训练时的模型保持一致。如果两个模型仅仅是定义时的名称有所不同（例如训练时定义为lstm，加载时的定义为lstm_layer），则可以通过改变模型名称的方式进行加载。

 

6. 对于小数据集，可以用hfd5的数据格式，大数据集(1000h以上)一般建议使用pickle格式。小数据集可以加载到内存中进行训练（dataset.py中的SpeechDatasetMem()），大数据集建议使用SpeechDataset()

 

7. **实验结果不好，可以从哪几个方向调整？**

      ①学习速率设置是否合理？何时开始learning rate decay比较合理?需要观察验证集loss曲线。

      ②如果是BLSTM模型，隐藏单元数目设置是否合理？与数据集大小有关。

      ③如果训练没有出现很大的波动，CTC权重可适当降低，或者不用CTC损失函数加权。

      ④改变神经网络模型。如加上VGG layer等

 

8. **训练有异常，从哪些方面检查？**

      ①Data/den_meta目录下，分母图生成是否异常？

      ②训练集、验证集的text_number转化是否有误？weight计算是否正常？若有，检查数据准备步骤是否有误。

      其他......
      
9. **如何开发新的egs?**

	开发一个新的egs，主要涉及到数据准备，神经网络训练的代码可以复用。数据准备方面，如果与训练集、测试集划分、特征提取、预处理有关的代码，可参考KALDI。WFST构建（包括T、L、G的生成与组合）的代码，可参考EESEN。CAT中的aishell和swbd可分别作为中英文数据集开发的蓝本。

10. **如何获得可复现的实验结果？**

	需要固定随机数种子。一个实验里可能用到的随机数有很多，例如：
	torch.manual_seed(123) 
	torch.cuda.manual_seed(123) 
	np.ranom.seed(123) 
	random.seed(123) 
	torch.backends.cudnn.deterministic=True

11. **中文数据集，测试时的字数目与实际字数目不匹配？**

      需检查字符编码是否一致。
      
12. **训练语言模型时报错Usage: optimize_alpha.pl alpha1 perplexity@alpha1 alpha2 perplexity@alpha2 alpha3 perplexity@alph3 at /xxx/xxx/kaldi/tools/kaldi_lm/optimize_alpha.pl line 23.**

      原因是调用了Kaldi中的脚本train_lm.sh进行语言模型的训练，而这个脚本默认语料文本至少10k句，held_out参数默认设置为10000，如果训练集的总句子数小于held_out的值就会出错。对于语料较少的情况，可以考虑将held_out改小。修改该参数后，还要注意将data目录下的文件全部删除，再运行run.sh才能成功。
    
13. **发现生成的data/dict_phn/lexicon_numbers.txt文件为空？**

      是因为生成此文件时，在词典lexicon.txt中存在非法音素，这些音素不包含在在units.txt音素集中。一般可能是lexicon.txt中出现了连续的多余空格，空格是没有被收录在units.txt中的，删去多余的空格即可。
   
14. **网络训练时，出现CUDA error：out of memory？**

      尝试改小batch size。
    
15. **解码时出错，log文件中提示latgen-faster command not found？**

      因为CAT安装中第一步，latgen-faster编译未成功。需要将latgen-faster加入kaldi的src中，并在Makefile文件中加上latgen-faster，之后在kaldi的src目录下make，才能成功编译。
    
16. **loss出现不降反增的情况？**
  
      检查网络的输出单元数与labels的维度是否匹配，不匹配时会出现学习错误的情况，与labels维度（即音素数量）保持一致即可。
    
17. **data/dict_phn/units.txt中出现了非音素集的字符？**
  
      检查词典的单词中是否有空格，如果有，处理脚本会将空格后部分的单词当作注音，将其每个字母当作新的音素，导致音素集引入更多其他的符号。
      
      
      
      
      
      
