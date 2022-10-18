# To Adapt or to Fine-tune: A Case Study on Abstractive Summarization
This repository contains code for paper "[To Adapt or to Fine-tune: A Case Study on Abstractive Summarization](https://arxiv.org/abs/2208.14559)" which appears in 2022 Chinese National Conference on Computational Linguistics (CCL).

## Required Packages
To install required packages, run the following command:

```
pip -r requirements.txt
```

## Data
The following dataset are used in our paper:

- `BookSum`
- `DialogSum`
- `NCLS`
- `SAMSum`
- `WikiLingua`
- `XL-Sum`

For more descriptions of the datasets and their uses, please refer to Sec 4.1/5.1/6.1 in the paper. 

You can download the dataset [here](https://drive.google.com/file/d/1uMnvXQAVUHXFd1gzlDEQjNFTu9-66X7h/view?usp=sharing). Unzip the data using the following command:
```
unzip ccl_data.zip 
```

## Training 

To train models, we use `accelerate` package which handles multi-GPU training. The following example is used to train a mBART on NCLS using 4 NVIDIA A100 GPUs:

```
accelerate launch code/run_mBART.py \
  --batch_size 12 \
  --epochs 100 \
  --model_type facebook/mbart-large-cc25 \
  --data_path path/to/ncls/dataset/ \
  --src_lang en_XX \
  --tgt_lang zh_CN \
  --output_dir path/to/checkpoint/directory/ \
  --seed set-a-random-seed-here \
  --fp16 \
  --prediction_file path/to/prediction/directory/model-output.txt \
  --reference_file path/to/prediction/directory/model-output.gold.txt \
  --checkpoint specify-checkpoint-name-here \
  --lr 1e-4 \
  --distributed \
  --max_input_len 1024 \
  --max_output_len 256
```

The following example is used to train mBART with adapters (`houlsby` variant with `reduction factor` being 2) on NCLS:
```
accelerate launch code/run_mBART_adapters.py \
  --batch_size 12 \
  --epochs 200 \
  --model_type facebook/mbart-large-cc25 \
  --data_path path/to/ncls/dataset/ \
  --src_lang en_XX \
  --tgt_lang zh_CN \
  --output_dir path/to/checkpoint/directory/ \
  --seed set-a-random-seed-here \
  --prediction_file path/to/prediction/directory/model-output.txt \
  --reference_file path/to/prediction/directory/model-output.gold.txt \
  --checkpoint specify-checkpoint-name-here \
  --lr 1e-4 \
  --fp16 \
  --save_adapter \
  --train_adapter \
  --distributed \
  --task_name mbart \
  --adapter_config houlsby \
  --adapter_reduction_factor 2 \
  --max_output_len 256 \
  --max_input_len 1024
```

For other experiments, simply change `data_path` and corresponding hyperparameters like `lr` and `adapter_reduction_factor`.

## Inference
The following example does inferencing using mBART on NCLS:
```
python code/run_mBART.py \
  --batch_size 12 \
  --model_type facebook/mbart-large-cc25 \
  --data_path path/to/ncls/dataset/ \
  --src_lang en_XX \
  --tgt_lang zh_CN \
  --output_dir path/to/checkpoint/directory/ \
  --seed specify-random-seed-here \
  --checkpoint checkpoint-name.pt \
  --prediction_file path/to/prediction/directory/model-output.txt \
  --reference_file path/to/prediction/directory/model-output.gold.txt \
  --lr 1e-4 \
  --max_input_len 1024 \
  --max_output_len 256 \
  --task test 
```

The following example does inferenceing using mBART with adapter on NCLS:

```
python code/run_mBART_adapters.py \
  --batch_size 12 \
  --model_type facebook/mbart-large-cc25 \
  --data_path path/to/ncls/dataset/ \
  --src_lang en_XX \
  --tgt_lang zh_CN \
  --output_dir path/to/checkpoint/directory/ \
  --seed specify-random-seed-here \
  --fp16 \
  --checkpoint checkpoint-name.pt \
  --prediction_file path/to/prediction/directory/model-output.txt \
  --reference_file path/to/prediction/directory/model-output.gold.txt \
  --lr 1e-4 \
  --save_adapter \
  --train_adapter \
  --task_name mbart \
  --adapter_config houlsby \
  --adapter_reduction_factor 2 \
  --inference_min_len 10 \
  --inference_max_len 256 \
  --max_output_len 256 \
  --task test
```

## Evaluation 
We use ROUGE score as the evaluation metric. The default ROUGE package we use in our code is [this one](https://github.com/huggingface/datasets/blob/main/metrics/rouge/rouge.py). With this package, we evaluated Chinese and Japanese on char-level. Since the original ROUGE is intended for English texts, we recommend using [this repo](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) for multilingual ROUGE evaluation. For more details, please refer to Sec 3.3 of the paper. 
## Citation
```
@inproceedings{zhao-chen-2022-adapt,
    title = "To Adapt or to Fine-tune: A Case Study on Abstractive Summarization",
    author = "Zhao, Zheng  and
      Chen, Pinzhen",
    booktitle = "Proceedings of the 21st Chinese National Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Nanchang, China",
    publisher = "Chinese Information Processing Society of China",
    url = "https://aclanthology.org/2022.ccl-1.73",
    pages = "824--835",
    language = "English",
}
```
