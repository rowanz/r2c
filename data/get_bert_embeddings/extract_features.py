# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import requests
import zipfile
import numpy as np

from data.get_bert_embeddings import modeling
from data.get_bert_embeddings import tokenization
import tensorflow as tf
import h5py
from tqdm import tqdm
from data.get_bert_embeddings.vcr_loader import data_iter, data_iter_test, convert_examples_to_features, input_fn_builder

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("name", 'bert', "The name to use")

flags.DEFINE_string("split", 'train', "The split to use")

flags.DEFINE_string("layers", "-2", "")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", 'uncased_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("endingonly", False, "Only use the ending")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

####

if not os.path.exists('uncased_L-12_H-768_A-12'):
    response = requests.get('https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip',
                            stream=True)
    with open('uncased_L-12_H-768_A-12.zip', "wb") as handle:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  # filter out keep-alive new chunks
                handle.write(chunk)
    with zipfile.ZipFile('uncased_L-12_H-768_A-12.zip') as zf:
        zf.extractall()

print("BERT HAS BEEN DOWNLOADED")
mypath = os.getcwd()
bert_config_file = os.path.join(mypath, 'uncased_L-12_H-768_A-12', 'bert_config.json')
vocab_file = os.path.join(mypath, 'uncased_L-12_H-768_A-12', 'vocab.txt')
# init_checkpoint = os.path.join(mypath, 'uncased_L-12_H-768_A-12', 'bert_model.ckpt')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, _) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


tf.logging.set_verbosity(tf.logging.INFO)

layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)
########################################

data_iter_ = data_iter if FLAGS.split != 'test' else data_iter_test
examples = [x for x in data_iter_(f'../{FLAGS.split}.jsonl',
                                 tokenizer=tokenizer,
                                 max_seq_length=FLAGS.max_seq_length,
                                 endingonly=FLAGS.endingonly)]
features = convert_examples_to_features(
    examples=[x[0] for x in examples], seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)
unique_id_to_ind = {}
for i, feature in enumerate(features):
    unique_id_to_ind[feature.unique_id] = i

############################ Tensorflow boilerplate
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    master=FLAGS.master,
    tpu_config=tf.contrib.tpu.TPUConfig(
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=FLAGS.init_checkpoint,
    layer_indexes=layer_indexes,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

# If TPU is not available, this will fall back to normal Estimator on CPU
# or GPU.
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    predict_batch_size=FLAGS.batch_size)

input_fn = input_fn_builder(
    features=features, seq_length=FLAGS.max_seq_length)

output_h5_qa = h5py.File(f'../{FLAGS.name}_answer_{FLAGS.split}.h5', 'w')
output_h5_qar = h5py.File(f'../{FLAGS.name}_rationale_{FLAGS.split}.h5', 'w')

if FLAGS.split != 'test':
    subgroup_names = [
        'answer0',
        'answer1',
        'answer2',
        'answer3',
        'rationale0',
        'rationale1',
        'rationale2',
        'rationale3',
    ]
else:
    subgroup_names = [
        'answer0',
        'answer1',
        'answer2',
        'answer3'] + [f'rationale{x}{y}' for x in range(4) for y in range(4)]

for i in range(len(examples) // len(subgroup_names)):
    output_h5_qa.create_group(f'{i}')
    output_h5_qar.create_group(f'{i}')


def alignment_gather(alignment, layer):
    reverse_alignment = [[i for i, x in enumerate(alignment) if x == j] for j in range(max(alignment) + 1)]
    output_embs = np.zeros((max(alignment) + 1, layer.shape[1]), dtype=np.float16)

    # Make sure everything is covered
    uncovered = np.zeros(max(alignment) + 1, dtype=np.bool)

    for j, trgs in enumerate(reverse_alignment):
        if len(trgs) == 0:
            uncovered[j] = True
        else:
            output_embs[j] = np.mean(layer[trgs], 0).astype(np.float16)
            things_to_fill = np.where(uncovered[:j])[0]
            if things_to_fill.shape[0] != 0:
                output_embs[things_to_fill] = output_embs[j]
                uncovered[:j] = False
    return output_embs


for result in tqdm(estimator.predict(input_fn, yield_single_examples=True)):
    ind = unique_id_to_ind[int(result["unique_id"])]

    text, ctx_alignment, choice_alignment = examples[ind]
    # just one layer for now
    layer = result['layer_output_0']
    ex2use = ind//len(subgroup_names)
    subgroup_name = subgroup_names[ind % len(subgroup_names)]

    group2use = (output_h5_qa if subgroup_name.startswith('answer') else output_h5_qar)[f'{ex2use}']
    alignment_ctx = [-1] + ctx_alignment

    if FLAGS.endingonly:
        # just a single span here
        group2use.create_dataset(f'answer_{subgroup_name}', data=alignment_gather(alignment_ctx, layer))
    else:
        alignment_answer = [-1] + [-1 for i in range(len(ctx_alignment))] + [-1] + choice_alignment
        group2use.create_dataset(f'ctx_{subgroup_name}', data=alignment_gather(alignment_ctx, layer))
        group2use.create_dataset(f'answer_{subgroup_name}', data=alignment_gather(alignment_answer, layer))
