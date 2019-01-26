import json
from collections import defaultdict

import tensorflow as tf
from tqdm import tqdm


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b, is_correct=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.is_correct = is_correct


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, is_correct):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.is_correct = is_correct


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn',
                        ]


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        # note, this is different because weve already tokenized
        tokens_a = example.text_a

        # tokens_b = example.text_b

        tokens_b = None
        if example.text_b:
            tokens_b = example.text_b

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                is_correct=example.is_correct))
    return features


##################################################################################################

def _fix_tokenization(tokenized_sent, obj_to_type, det_hist=None):
    if det_hist is None:
        det_hist = {}
    else:
        det_hist = {k: v for k, v in det_hist.items()}

    obj2count = defaultdict(int)
    # Comment this in if you want to preserve stuff from the earlier rounds.
    for v in det_hist.values():
        obj2count[v.split('_')[0]] += 1

    new_tokenization = []
    for i, tok in enumerate(tokenized_sent):
        if isinstance(tok, list):
            for int_name in tok:
                if int_name not in det_hist:
                    if obj_to_type[int_name] == 'person':
                        det_hist[int_name] = GENDER_NEUTRAL_NAMES[obj2count['person'] % len(GENDER_NEUTRAL_NAMES)]
                    else:
                        det_hist[int_name] = obj_to_type[int_name]
                    obj2count[obj_to_type[int_name]] += 1
                new_tokenization.append(det_hist[int_name])
        else:
            new_tokenization.append(tok)
    return new_tokenization, det_hist


def fix_item(item, answer_label=None, rationales=True):
    if rationales:
        assert answer_label is not None
        ctx = item['question'] + item['answer_choices'][answer_label]
    else:
        ctx = item['question']

    q_tok, hist = _fix_tokenization(ctx, item['objects'])
    choices = item['rationale_choices'] if rationales else item['answer_choices']
    a_toks = [_fix_tokenization(choice, obj_to_type=item['objects'], det_hist=hist)[0] for choice in choices]
    return q_tok, a_toks


def retokenize_with_alignment(span, tokenizer):
    tokens = []
    alignment = []
    for i, tok in enumerate(span):
        for token in tokenizer.basic_tokenizer.tokenize(tok):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                tokens.append(sub_token)
                alignment.append(i)
    return tokens, alignment


def process_ctx_ans_for_bert(ctx_raw, ans_raw, tokenizer, counter, endingonly, max_seq_length, is_correct):
    """
    Processes a Q/A pair for BERT
    :param ctx_raw:
    :param ans_raw:
    :param tokenizer:
    :param counter:
    :param endingonly:
    :param max_seq_length:
    :param is_correct:
    :return:
    """
    context = retokenize_with_alignment(ctx_raw, tokenizer)
    answer = retokenize_with_alignment(ans_raw, tokenizer)

    if endingonly:
        take_away_from_ctx = len(answer[0]) - max_seq_length + 2
        if take_away_from_ctx > 0:
            answer = (answer[0][take_away_from_ctx:],
                      answer[1][take_away_from_ctx:])

        return InputExample(unique_id=counter, text_a=answer[0], text_b=None,
                            is_correct=is_correct), answer[1], None

    len_total = len(context[0]) + len(answer[0]) + 3
    if len_total > max_seq_length:
        take_away_from_ctx = min((len_total - max_seq_length + 1) // 2, max(len(context) - 32, 0))
        take_away_from_answer = len_total - max_seq_length + take_away_from_ctx
        context = (context[0][take_away_from_ctx:],
                   context[1][take_away_from_ctx:])
        answer = (answer[0][take_away_from_answer:],
                  answer[1][take_away_from_answer:])

        print("FOR Q{} A {}\nLTotal was {} so take away {} from ctx and {} from answer".format(
            ' '.join(context[0]), ' '.join(answer[0]), len_total, take_away_from_ctx,
            take_away_from_answer), flush=True)
    assert len(context[0]) + len(answer[0]) + 3 <= max_seq_length

    return InputExample(unique_id=counter,
                        text_a=context[0],
                        text_b=answer[0], is_correct=is_correct), context[1], answer[1]


def data_iter(data_fn, tokenizer, max_seq_length, endingonly):
    counter = 0
    with open(data_fn, 'r') as f:
        for line_no, line in enumerate(tqdm(f)):
            item = json.loads(line)
            q_tokens, a_tokens = fix_item(item, rationales=False)
            qa_tokens, r_tokens = fix_item(item, answer_label=item['answer_label'], rationales=True)

            for (name, ctx, answers) in (('qa', q_tokens, a_tokens), ('qar', qa_tokens, r_tokens)):
                for i in range(4):
                    is_correct = item['answer_label' if name == 'qa' else 'rationale_label'] == i

                    yield process_ctx_ans_for_bert(ctx, answers[i], tokenizer, counter=counter, endingonly=endingonly,
                                                   max_seq_length=max_seq_length, is_correct=is_correct)
                    counter += 1


def data_iter_test(data_fn, tokenizer, max_seq_length, endingonly):
    """ Essentially this needs to be a bit separate from data_iter because we don't know which answer is correct."""
    counter = 0
    with open(data_fn, 'r') as f:
        for line_no, line in enumerate(tqdm(f)):
            item = json.loads(line)
            q_tokens, a_tokens = fix_item(item, rationales=False)

            # First yield the answers
            for i in range(4):
                yield process_ctx_ans_for_bert(q_tokens, a_tokens[i], tokenizer, counter=counter, endingonly=endingonly,
                                               max_seq_length=max_seq_length, is_correct=False)
                counter += 1

            for i in range(4):
                qa_tokens, r_tokens = fix_item(item, answer_label=i, rationales=True)
                for r_token in r_tokens:
                    yield process_ctx_ans_for_bert(qa_tokens, r_token, tokenizer, counter=counter,
                                                   endingonly=endingonly,
                                                   max_seq_length=max_seq_length, is_correct=False)
                    counter += 1
