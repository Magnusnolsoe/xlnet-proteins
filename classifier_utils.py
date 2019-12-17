from absl import flags

import re
import numpy as np

import tensorflow as tf
from data_utils import EOP_ID

FLAGS = flags.FLAGS

SEG_ID   = 0
SEG_ID_EOP = 1
SEG_ID_PAD = 4

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def truncate(tokens, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens)
    if total_length <= max_length:
      break
    tokens.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                              tokenize_fn):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[1] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  if label_list is not None:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokenized = tokenize_fn(example.text)

  truncate(tokenized, max_seq_length-1)

  tokens = []
  segment_ids = []
  for token in tokenized:
    tokens.append(token)
    segment_ids.append(SEG_ID)

  tokens.append(EOP_ID)
  segment_ids.append(SEG_ID_EOP)

  input_ids = tokens

  # The mask has 0 for real tokens and 1 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [0] * len(input_ids)

  # Zero-pad up to the sequence length.
  if len(input_ids) < max_seq_length:
    delta_len = max_seq_length - len(input_ids)
    input_ids = [0] * delta_len + input_ids
    input_mask = [1] * delta_len + input_mask
    segment_ids = [SEG_ID] * delta_len + segment_ids

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if label_list is not None:
    label_id = label_map[example.label]
  else:
    label_id = example.label
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: {} (id = {})".format(example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature



