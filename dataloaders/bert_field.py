from typing import Dict, List, Optional
import textwrap

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util
import numpy
TokenList = List[TokenType]  # pylint: disable=invalid-name


# This will work for anything really
class BertField(SequenceField[Dict[str, torch.Tensor]]):
    """
    A class representing an array, which could have arbitrary dimensions.
    A batch of these arrays are padded to the max dimension length in the batch
    for each dimension.
    """
    def __init__(self, tokens: List[Token], embs: numpy.ndarray, padding_value: int = 0,
            token_indexers=None) -> None:
        self.tokens = tokens
        self.embs = embs
        self.padding_value = padding_value

        if len(self.tokens) != self.embs.shape[0]:
            raise ValueError("The tokens you passed into the BERTField, {} "
                             "aren't the same size as the embeddings of shape {}".format(self.tokens, self.embs.shape))
        assert len(self.tokens) == self.embs.shape[0]

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)


    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        num_tokens = padding_lengths['num_tokens']

        new_arr = numpy.ones((num_tokens, self.embs.shape[1]),
                             dtype=numpy.float32) * self.padding_value
        new_arr[:self.sequence_length()] = self.embs

        tensor = torch.from_numpy(new_arr)
        return {'bert': tensor}

    @overrides
    def empty_field(self):
        return BertField([], numpy.array([], dtype="float32"),padding_value=self.padding_value)

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.
        return util.batch_tensor_dicts(tensor_list)


    def __str__(self) -> str:
        return f"BertField: {self.tokens} and  {self.embs.shape}."
