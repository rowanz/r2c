"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
# get gt labels
labels = {
    'val_rationale': [],
    'test_rationale': [],
    'val_answer': [],
    'test_answer': [],
}
for split in ('val', 'test'):
    with open(f'/home/rowan/code/vswagmodels/data/{split}.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            labels[f'{split}_answer'].append(item['answer_label'])
            labels[f'{split}_rationale'].append(item['rationale_label'])
for k in labels:
    labels[k] = np.array(labels[k])

folders = [
    'flagship_answer',
    'no_reasoning_rationale',
    'no_reasoning_answer',
    'flagship_rationale',
    'no_question_answer',
    'no_vision_answer',
    'no_vision_rationale',
    'bottom_up_top_down_glove_rationale',
    'bottom_up_top_down_rationale',
    'bottom_up_top_down_glove_answer',
    'no_question_rationale',
    'vqa_baseline_glove_answer',
    'vqa_baseline_glove_rationale',
    'bottom_up_top_down_answer',
    'glove_answer',
    'glove_rationale',
    'mutan_glove_answer',
    'mutan_glove_rationale',
    'mlb_glove_answer',
    'mlb_glove_rationale',
    'esim_answer',
    'esim_rationale',
    'lstm_answer',
    'lstm_rationale',
]

folder_to_preds = {folder: (np.load(os.path.join('/home/rowan/datasets3/vswagmodels/', folder, 'valpreds.npy')),
                            np.load(os.path.join('/home/rowan/datasets3/vswagmodels/', folder, 'testpreds.npy'))) for folder in folders}

def _softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(1)[:,None]

log_folders = [
    'bert_answer',
    'bert_rationale',
    'bert_answer_ending',
    'bert_rationale_ending',
]
for folder in log_folders:
    new_name = {'bert_answer_ending': 'bert_ending_answer', 'bert_rationale_ending': 'bert_ending_rationale'}.get(folder, folder)
    folder_to_preds[new_name] = (np.exp(np.load(os.path.join('/home/rowan/datasets3/vswagmodels/', folder, 'val-logprobs.npy'))),
                               np.exp(np.load(os.path.join('/home/rowan/datasets3/vswagmodels/', folder, 'test-logprobs.npy'))))

# sanity check
for x, (y, z) in folder_to_preds.items():
    assert np.abs(np.mean(y.sum(1)) - 1.0).sum() < 0.0001
    assert np.abs(np.mean(z.sum(1)) - 1.0).sum() < 0.0001

base_folders = sorted(set(['_'.join(x.split('_')[:-1]) for x in folder_to_preds]))
for folder in base_folders:
    print("\n\n\nFor {}".format(folder), flush=True)
    for split_id, split_name in enumerate(['val', 'test']):
        print("{}".format(split_name), flush=True)
        # Answer
        answer_hits = folder_to_preds[folder + '_answer'][split_id].argmax(1) == labels[split_name + '_answer']
        rationale_hits = folder_to_preds[folder + '_rationale'][split_id].argmax(1) == labels[split_name + '_rationale']

        print(" Answer acc: {:.3f}".format(np.mean(answer_hits)), flush=True)
        print(" Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
        print(" Joint acc: {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)
