"""
Dataloaders for VCR
"""
import json
import os

import numpy as np
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']


# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }

def _fix_tokenization(tokenized_sent, bert_embs, old_det_to_new_ind, obj_to_type, token_indexers, pad_ind=-1):
    """
    Turn a detection list into what we want: some text, as well as some tags.
    :param tokenized_sent: Tokenized sentence with detections collapsed to a list.
    :param old_det_to_new_ind: Mapping of the old ID -> new ID (which will be used as the tag)
    :param obj_to_type: [person, person, pottedplant] indexed by the old labels
    :return: tokenized sentence
    """

    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind
                    ))
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_ind))

    text_field = BertField([Token(x[0]) for x in new_tokenization_with_tags],
                           bert_embs,
                           padding_value=0)
    tags = SequenceLabelField([x[1] for x in new_tokenization_with_tags], text_field)
    return text_field, tags


class VCR(Dataset):
    def __init__(self, split, mode, only_use_relevant_dets=True, add_image_as_a_box=True, embs_to_load='bert_da',
                 conditioned_answer_choice=0):
        """

        :param split: train, val, or test
        :param mode: answer or rationale
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the question and answer.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects.
        :param embs_to_load: Which precomputed embeddings to load.
        :param conditioned_answer_choice: If you're in test mode, the answer labels aren't provided, which could be
                                          a problem for the QA->R task. Pass in 'conditioned_answer_choice=i'
                                          to always condition on the i-th answer.
        """
        self.split = split
        self.mode = mode
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        self.add_image_as_a_box = add_image_as_a_box
        self.conditioned_answer_choice = conditioned_answer_choice

        with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        if split not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))

        if mode not in ('answer', 'rationale'):
            raise ValueError("split must be answer or rationale")

        self.token_indexers = {'elmo': ELMoTokenCharactersIndexer()}
        self.vocab = Vocabulary()

        with open(os.path.join(os.path.dirname(VCR_ANNOTS_DIR), 'dataloaders', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        self.embs_to_load = embs_to_load
        self.h5fn = os.path.join(VCR_ANNOTS_DIR, f'{self.embs_to_load}_{self.mode}_{self.split}.h5')
        print("Loading embeddings from {}".format(self.h5fn), flush=True)

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy['mode'] = 'answer'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['mode', 'split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split='test', mode='answer', **kwargs)] + [
            cls(split='test', mode='rationale', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item):
        """
        We might want to use fewer detectiosn so lets do so.
        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        # if self.split == 'test':
        #     raise ValueError("blind test mode not supported quite yet")
        item = deepcopy(self.items[index])

        ###################################################################
        # Load questions and answers
        if self.mode == 'rationale':
            conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
            item['question'] += item['answer_choices'][conditioned_label]

        answer_choices = item['{}_choices'.format(self.mode)]
        dets2use, old_det_to_new_ind = self._get_dets_to_use(item)

        ###################################################################
        # Load in BERT. We'll get contextual representations of the context and the answer choices
        # grp_items = {k: np.array(v, dtype=np.float16) for k, v in self.get_h5_group(index).items()}
        with h5py.File(self.h5fn, 'r') as h5:
            grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}

        # Essentially we need to condition on the right answer choice here, if we're doing QA->R. We will always
        # condition on the `conditioned_answer_choice.`
        condition_key = self.conditioned_answer_choice if self.split == "test" and self.mode == "rationale" else ""

        instance_dict = {}
        if 'endingonly' not in self.embs_to_load:
            questions_tokenized, question_tags = zip(*[_fix_tokenization(
                item['question'],
                grp_items[f'ctx_{self.mode}{condition_key}{i}'],
                old_det_to_new_ind,
                item['objects'],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1
            ) for i in range(4)])
            instance_dict['question'] = ListField(questions_tokenized)
            instance_dict['question_tags'] = ListField(question_tags)

        answers_tokenized, answer_tags = zip(*[_fix_tokenization(
            answer,
            grp_items[f'answer_{self.mode}{condition_key}{i}'],
            old_det_to_new_ind,
            item['objects'],
            token_indexers=self.token_indexers,
            pad_ind=0 if self.add_image_as_a_box else -1
        ) for i, answer in enumerate(answer_choices)])

        instance_dict['answers'] = ListField(answers_tokenized)
        instance_dict['answer_tags'] = ListField(answer_tags)
        if self.split != 'test':
            instance_dict['label'] = LabelField(item['{}_label'.format(self.mode)], skip_indexing=True)
        instance_dict['metadata'] = MetadataField({'annot_id': item['annot_id'], 'ind': index, 'movie': item['movie'],
                                                   'img_fn': item['img_fn'],
                                                   'question_number': item['question_number']})

        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(os.path.join(VCR_IMAGES_DIR, item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        ###################################################################
        # Load boxes.
        with open(os.path.join(VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i])
                          for i in dets2use])

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        instance_dict['segms'] = ArrayField(segms, padding_value=0)
        instance_dict['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        instance = Instance(instance_dict)
        instance.index_fields(self.vocab)
        return image, instance


def collate_fn(data, to_gpu=False):
    """Creates mini-batch tensors
    """
    images, instances = zip(*data)
    images = torch.stack(images, 0)
    batch = Batch(instances)
    td = batch.as_tensor_dict()
    if 'question' in td:
        td['question_mask'] = get_text_field_mask(td['question'], num_wrapping_dims=1)
        td['question_tags'][td['question_mask'] == 0] = -2  # Padding

    td['answer_mask'] = get_text_field_mask(td['answers'], num_wrapping_dims=1)
    td['answer_tags'][td['answer_mask'] == 0] = -2

    td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
    td['images'] = images

    # Deprecated
    # if to_gpu:
    #     for k in td:
    #         if k != 'metadata':
    #             td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
    #             non_blocking=True)
 
    # # No nested dicts
    # for k in sorted(td.keys()):
    #     if isinstance(td[k], dict):
    #         for k2 in sorted(td[k].keys()):
    #             td['{}_{}'.format(k, k2)] = td[k].pop(k2)
    #         td.pop(k)

    return td


class VCRLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, batch_size=3, num_workers=6, num_gpus=3, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            shuffle=data.is_train,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, to_gpu=False),
            drop_last=data.is_train,
            pin_memory=False,
            **kwargs,
        )
        return loader

# You could use this for debugging maybe
# if __name__ == '__main__':
#     train, val, test = VCR.splits()
#     for i in range(len(train)):
#         res = train[i]
#         print("done with {}".format(i))
