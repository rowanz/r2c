"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""
import numpy as np
import json
import os

# Variables for groundtruth labels for the validation set
labels = {
    'val_rationale': [],
    'val_answer': []
}

# Obtain groundtruth labels 
with open('./data/val.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        labels[f'{val_answer'].append(item['answer_label'])
        labels[f'{val_rationale'].append(item['rationale_label'])
for k in labels:
    labels[k] = np.array(labels[k])

# Load the predicted answers (Q->A) and rationales (QA->R)
ans = np.load('/DATA/radhika/images/vcrimg/saves/flagship_answer/valpreds.npy')
rat = np.load('/DATA/radhika/images/vcrimg/saves/flagship_rationale/valpreds.npy')

# Accuracy in question-answering, answer justification and joint accuracy
answer_hits = ans.argmax(1) == labels['val_answer']
rationale_hits = rat.argmax(1) == labels['val_rationale']

print(" Answer acc: {:.3f}".format(np.mean(answer_hits)), flush=True)
print(" Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
print(" Joint acc: {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)

