# Extracting BERT representations

Replicating my results with R2C requires precomputing BERT representations of the dataset. These representations are really expensive to precompute, so doing so saves a lot of time.

You can download them here:


## Extracting BERT representations yourself

If you want to do so yourself, create a condaenv with tensorflow 1.11 installed. Here we'll call it `bert`. You can use
the following command to compute BERT representations of `train.jsonl` in the `data/` folder:

```
source activate bert
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export PYTHONPATH=/user/home/rowan/code/r2c
export CUDA_VISIBLE_DEVICES=0

python extract_features.py --name bert --split=train
```

## Domain adaptation

In my early experiments, I found domain adaptation to be important with BERT, mainly because VCR is quite different than books/wikipedia style-wise. So, for the results in the paper, I performed domain adaptation as follows:

First, I used the following script to create `pretrainingdata.tfrecord`:
```
source activate bert
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export PYTHONPATH=/user/home/rowan/code/r2c

python create_pretraining_data.py
```

Then, I trained BERT on that, using

```
export CUDA_VISIBLE_DEVICES=0

python pretrain_on_vcr.py --do_train 
```

This creates a folder called `bert-pretrain`. Now, extract the features as follows.

```
python extract_features.py --name bert_da --init_checkpoint bert-pretrain/model.ckpt-53230 --split=train
```

