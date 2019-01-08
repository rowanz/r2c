# models

This folder is for `r2c` models. They broadly follow the allennlp configuration format. If you want r2c, you'll want to look at `multiatt`.

Here's how you can replicate my val results. Run the command(s) below. First, you might want to make your GPUs available. When I ran these experiments I used

`source activate r2c && export LD_LIBRARY_PATH=/usr/local/cuda-9.0/ && export PYTHONPATH=/home/rowan/code/r2c && export CUDA_VISIBLE_DEVICES=0,1,2`

## Question Answering:

```
python train.py -params multiatt/default.json -folder saves/flagship_answer 
```

## Answer justification:
```
python train.py -params multiatt/default.json -folder saves/flagship_rationale -rationale
```

