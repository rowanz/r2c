# Data

Obtain the dataset by visiting [visualcommonsense.com/download.html](https://visualcommonsense.com/download.html). 
 - Extract the images somewhere. I put them in a different directory, `/home/rowan/datasets2/vcr1/vcr1images` and added a symlink in this (`data`): `ln -s /home/rowan/datasets2/vcr1/vcr1images`
 - Put `train.jsonl`, `val.jsonl`, and `test.jsonl` in here (`data`).
 
You can also put the dataset somewhere else, you'll just need to update `config.py` (in the main directory) accordingly.
```
unzip vcr1annots.zip
```

# Precomputed representations
Running R2c requires computed bert representations in this folder. Warning: these files are quite large. You have two options to generate these:

1. (recommended) download them from :
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_test.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_test.h5`
2. You can use the script in the folder `get_bert_embeddings` to precompute BERT representations for all sentences.

