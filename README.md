Our code is based on  the Transformers repo: https://github.com/huggingface/transformers/ Version 3.4.0

## Dialogue-oriented pre-training on Wikipedia
### Dataset
* Our data sampling code is in examples/data_sampling

    Firstly, Please download datasets to directory "mydata"

    English
    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

    Chinese
    https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2


* Sampling code

    MyProcess_wiki.py/zhMyProcess_wiki.py are to generate articles from raw English/Chinese Wikipedia dataset.
    
    zhwiki_tradition2simplify.py is to convert traditional Chinese into simplified version.

    data_sampling_3.py/zh_data_sampling_3.py is to sample from each article for three tasks.
    
    gather_wikipretrain_data.py is to gather samples from each article to generate final pre-trainig datasets of the three tasks.

* Our pre-training datasets are uploaded to https://drive.google.com/drive/folders/1v8HYbE6A28GWT19lk6pC4xi0cgFraxc_?usp=sharing
### Pre-training
* Our pre-training code for English/Chinese & BERT/ELECTRA is in examples/wiki_pretrain
   
    Please download pre-trainig datasets to wikipretrain_v3, then you can directly run
    
    <code>python run_bert_wikipretrain_sptoken_sot_3.py</code>  
    
    After pre-training, Dialog-BERT model will be saved in --output_dir.

* Our pre-trained Dialog-PrLM models are uploaded to https://drive.google.com/drive/folders/1wGRQMjMXzhKEWqx1-Q_pCB746YglxMPX?usp=sharing, where
    
    Dialog-BERT(en) is in "wikipretrain_v3";  Dialog-BERT(zh) is in "zh_wikipretrain_v3";
    Dialog-ELECTRA(en) is in "electra_base_wikipretrain_v3"; Dialog-ELECTRA(zh) is in "electra_base_zh_wikipretrain_v3"
    
## Response selection task
### Dataset
Please download datasets to the corresponding directory under "mydata"

E-commerce
https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view?usp=sharing.

Ubuntu
https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntudata.zip?dl=0

Douban
https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0&file_subpath=%2FDoubanConversaionCorpus

### Fine-tuning
* Our fine-tuning code on Dialog-PrLM is in examples/response_selection/RS_finetune
    
    Please put Dialog-BERT model into --bert_model, and choose task --task_name. Then run
    
    <code>python _old_run_response_selection_wikipretrain_RSmatching2_v3.py</code>  
    
    After fine-tuning, the model will be saved in --output_dir.

<!--
* Our pre-trained Dialog-PrLM models are uploaded to
-->
### Multi-task learning
* Our multi-task learning code on PrLM/Dialog-PrLM is in examples/response_selection/RS_multitask

    Please put Dialog-BERT model into --bert_model, and choose task --task_name. Then run
    
    <code>python run_bert_RSmultitask_v2.py</code>  
    
    After multi-task learning, the model will be saved in --output_dir.

    The run_bert_RSmultitask_v2.py/run_bert_RSmultitask_v2_electra.py are for Dialog-BERT/Dialog-ELECTRA; 
    
    The run_bert_RSmultitask_v2_baseline.py/run_bert_RSmultitask_v2_electra_baseline.py are for BERT/ELECTRA.
    
<!--
* Our pre-trained Dialog-PrLM models are uploaded to
-->
## Reference
 
If you use this code please cite our paper:
```
@article{xu2021dialogue,
  title={Dialogue-oriented Pre-training},
  author={Xu, Yi and Zhao, Hai},
  journal={ACL2021 Findings},
  year={2021}
}
```
