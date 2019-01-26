"""
Evaluation script for the leaderboard
"""
import argparse
import logging
import multiprocessing

import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.nn.util import device_mapping
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d

from dataloaders.vcr import VCR, VCRLoader
from utils.pytorch_misc import time_batch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

#################################
#################################
######## Data loading stuff
#################################
#################################

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    default='multiatt/default.json',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-answer_ckpt',
    dest='answer_ckpt',
    default='saves/flagship_answer/best.th',
    help='Answer checkpoint',
    type=str,
)
parser.add_argument(
    '-rationale_ckpt',
    dest='rationale_ckpt',
    default='saves/flagship_rationale/best.th',
    help='Rationale checkpoint',
    type=str,
)
parser.add_argument(
    '-output',
    dest='output',
    default='submission.csv',
    help='Output CSV file to save the predictions to',
    type=str,
)

args = parser.parse_args()
params = Params.from_file(args.params)

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")


def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        td[k] = {k2: v.cuda(async=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
            async=True)
    return td


num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2 * NUM_GPUS) - 1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}

vcr_modes = VCR.eval_splits(embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                            only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))
probs_grp = []
ids_grp = []
for (vcr_dataset, mode_long) in zip(vcr_modes, ['answer'] + [f'rationale_{i}' for i in range(4)]):
    mode = mode_long.split('_')[0]

    test_loader = VCRLoader.from_dataset(vcr_dataset, **loader_params)

    # Load the params again because allennlp will delete them... ugh.
    params = Params.from_file(args.params)
    print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), mode), flush=True)
    model = Model.from_params(vocab=vcr_dataset.vocab, params=params['model'])
    for submodule in model.detector.backbone.modules():
        if isinstance(submodule, BatchNorm2d):
            submodule.track_running_stats = False

    model_state = torch.load(getattr(args, f'{mode}_ckpt'), map_location=device_mapping(-1))
    model.load_state_dict(model_state)

    model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
    model.eval()

    test_probs = []
    test_ids = []
    for b, (time_per_batch, batch) in enumerate(time_batch(test_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            test_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            test_ids += [m['annot_id'] for m in batch['metadata']]
        if (b > 0) and (b % 10 == 0):
            print("Completed {}/{} batches in {:.3f}s".format(b, len(test_loader), time_per_batch * 10), flush=True)

    probs_grp.append(np.concatenate(test_probs, 0))
    ids_grp.append(test_ids)

################################################################################
# This is the part you'll care about if you want to submit to the leaderboard!
################################################################################

# Double check the IDs are in the same order for everything
assert [x == ids_grp[0] for x in ids_grp]

probs_grp = np.stack(probs_grp, 1)
# essentially probs_grp is a [num_ex, 5, 4] array of probabilities. The 5 'groups' are
# [answer, rationale_conditioned_on_a0, rationale_conditioned_on_a1,
#          rationale_conditioned_on_a2, rationale_conditioned_on_a3].
# We will flatten this to a CSV file so it's easy to submit.
group_names = ['answer'] + [f'rationale_conditioned_on_a{i}' for i in range(4)]
probs_df = pd.DataFrame(data=probs_grp.reshape((-1, 20)),
                        columns=[f'{group_name}_{i}' for group_name in group_names for i in range(4)])
probs_df['annot_id'] = ids_grp[0]
probs_df = probs_df.set_index('annot_id', drop=True)
probs_df.to_csv(args.output)
