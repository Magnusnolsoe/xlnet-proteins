from absl import flags

import re
import numpy as np
import collections

import tensorflow as tf
from data_utils import EOP_ID

FLAGS = flags.FLAGS

SEG_ID     = 0
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
               is_eop,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_eop = is_eop
    self.is_real_example = is_real_example  


def convert_single_example(ex_index, example, label_list, max_seq_length,
                              tokenize_fn):
  """Converts a single `InputExample` into a list of `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[1] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_eop=False,
        is_real_example=False)

  if label_list is not None:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  
  tokenized = tokenize_fn(example.text)
  protein_len = len(tokenized)

  assert protein_len > 0

  prot_is_divisable = protein_len % max_seq_length == 0 

  # Segment protein 
  _ProtSpan = collections.namedtuple(
                    "ProtSpan", ["start", "length"])
  prot_spans = []
  start_offset = 0
  while start_offset < protein_len:
    length = min(max_seq_length, protein_len-start_offset)
    prot_spans.append(_ProtSpan(start=start_offset, length=length))
    start_offset += length + 1
  
  if prot_is_divisable:
    prot_spans.append(_ProtSpan(start=protein_len, length=1))

  features = []
  for (prot_span_index, prot_span) in enumerate(prot_spans):

    start = prot_span.start
    end = prot_span.start + prot_span.length
    if prot_is_divisable:
      is_eop = prot_span.start + prot_span.length > protein_len
    else:
      is_eop = prot_span.start + prot_span.length == protein_len

    if prot_is_divisable and is_eop:
      tokens = []
      segment_ids = []
    else:
      tokens = tokenized[start : end]
      segment_ids = [SEG_ID] * prot_span.length
      
    if is_eop:
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
      segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

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
      tf.logging.info("is_eop: {}".format(is_eop))
      tf.logging.info("label: {} (id = {})".format(example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_eop=is_eop,
        label_id=label_id)
    
    features.append(feature)
  
  return features



