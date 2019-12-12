import os
import collections
import tensorflow as tf
import xlnet
import numpy as np
import pickle
import time

from absl import app, flags
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids
from model_utils import init_from_checkpoint


# I/O paths
flags.DEFINE_string("model_config_path", default=None,
    help="Path to model configuration file")
flags.DEFINE_string("init_checkpoint", default=None,
    help="checkpoint path for initializing the model. "
        "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="",
    help="Output dir for TF records.")
flags.DEFINE_string("data_dir", default="",
    help="Directory for input data.")

flags.DEFINE_integer("batch_size", default=1,
    help="Size of mini-batch")
flags.DEFINE_integer("max_seq_length", default=128,
    help="The maximum sequence length.")

# XLNet run configs
flags.DEFINE_bool("use_tpu", default=False,
    help="Whether or not to us TPU")
flags.DEFINE_bool("use_bfloat16", default=False,
    help="Whether or not to use bfloat16")
flags.DEFINE_float("dropout", default=0,
    help="Dropout rate")
flags.DEFINE_float("dropatt", default=0,
    help="Dropout rate used in attention")
flags.DEFINE_enum("init", default="normal",
        enum_values=["normal", "uniform"],
    help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
    help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
    help="Initialization std when init is uniform.")
flags.DEFINE_integer("clamp_len", default=-1,
    help="Clamp length")
FLAGS = flags.FLAGS


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text: string. The untokenized text of the sequence.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        if len(line) == 0: continue
        lines.append(line)
      return lines

  @classmethod
  def _read_txt(cls, input_file):
    """Reads a /n separated value file"""
    with tf.gfile.Open(input_file, "r") as f:
      lines = []
      for line in f:
        if len(line) == 0: continue
        lines.append(line)
      return lines

class SubLocProcessor(DataProcessor):
  def __init__(self):
    self.train_file = "train.txt"
    self.test_file = "test.txt"

  def get_train_examples(self, data_dir):
    return self._create_examples(
      self._read_txt(os.path.join(data_dir, self.train_file)), "train")

  def get_test_examples(self, data_dir):
    return self._create_examples(
      self._read_txt(os.path.join(data_dir, self.test_file)), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
  def _create_examples(self, lines, set_type):
    """Creates examples for the training and test sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)

      line = line.split(":")

      seq = line[0].strip()
      label = line[1].strip()

      examples.append(
          InputExample(guid=guid, text=seq, label=label))
    return examples

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenize_fn, output_file,
    num_passes=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  # do not create duplicated records
  if tf.gfile.Exists(output_file):
    tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
    return

  tf.logging.info("Create new tfrecord {}.".format(output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  if num_passes > 1:
    examples *= num_passes

  total_examples = 0

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example {} of {}".format(ex_index,
                                                        len(examples)))

    feature_list = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenize_fn)
    
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    for feature in feature_list:
      features = collections.OrderedDict()
      features["prot_id"] = create_int_feature([feature.prot_id])
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_float_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      if label_list is not None:
        features["label_ids"] = create_int_feature([feature.label_id])
      else:
        features["label_ids"] = create_float_feature([float(feature.label_id)])
      features["is_eop"] = create_int_feature(
          [int(feature.is_eop)])
      features["is_real_example"] = create_int_feature(
          [int(feature.is_real_example)])
      total_examples = total_examples + 1

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
  writer.close()
  return total_examples



def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
            example[name] = t

    return example

def get_dataset(input_file, seq_length, batch_size):
    """The actual input function."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_eop": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        "prot_id": tf.FixedLenFeature([], tf.int64),
    }

    tf.logging.info("Input tfrecord file {}".format(input_file))

    d = tf.data.TFRecordDataset(input_file)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=False))

    return d


def get_basename(seq_len, split):
    return "len-{}.{}.tf_record".format(seq_len, split)

def main(_):
    
    assert tf.gfile.Exists(FLAGS.init_checkpoint)

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    
    processor = SubLocProcessor()

    labels = processor.get_labels()
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    test_examples = processor.get_test_examples(FLAGS.data_dir)

    train_file_path = os.path.join(FLAGS.output_dir,
        get_basename(FLAGS.max_seq_length, "train"))
    test_file_path = os.path.join(FLAGS.output_dir,
        get_basename(FLAGS.max_seq_length, "test"))
    
    def tokenize_fn(text):
        text = preprocess_text(text)
        return encode_ids(text)

    # Create TF-Record for train examples
    file_based_convert_examples_to_features(train_examples, labels,
        FLAGS.max_seq_length, tokenize_fn, train_file_path)

    # Create TF-Record for test examples
    file_based_convert_examples_to_features(test_examples, labels,
        FLAGS.max_seq_length, tokenize_fn, test_file_path)

    train_set = get_dataset(train_file_path, FLAGS.max_seq_length, FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    example = train_iter.get_next()

    inp = tf.transpose(example["input_ids"], [1, 0])
    seg_id = tf.transpose(example["segment_ids"], [1, 0])
    inp_mask = tf.transpose(example["input_mask"], [1, 0])

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(False, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)

    output = xlnet_model.get_sequence_output()

    init_from_checkpoint(FLAGS)

    fetches = [output, example]

    with tf.Session() as sess:
        D = []
        T = []
        sess.run(tf.global_variables_initializer())
        try:
            prev_id = None
            d = []
            start = time.time()
            while True:
                fetched = sess.run(fetches)
                xlnet_out = np.squeeze(fetched[0], axis=1)
                inputs = fetched[1]
                seg_ids = inputs["segment_ids"][0]
                _id = inputs["prot_id"][0]
                target = inputs["label_ids"][0]
                idx = np.arange(np.sum(seg_ids == 0))
                selected = np.take(xlnet_out, indices=idx, axis=0)

                if _id == prev_id or prev_id is None:
                    d.append(selected)
                else:
                    D.append(np.concatenate(d))
                    T.append(target)
                    d = []
                    if len(D) % 10 == 0:
                        end = time.time()
                        tf.logging.info("Time {}".format((end-start)/60))
                        tf.logging.info("Processed {} proteins".format(len(D)))
                        start = end
                prev_id = _id
                    
        except tf.errors.OutOfRangeError:
            D.append(np.concatenate(d))
            T.append(target)
            with tf.gfile.Open(os.path.join(FLAGS.output_dir, "embeddings.p"), "wb") as fp:
              pickle.dump(D, fp)
            with tf.gfile.Open(os.path.join(FLAGS.output_dir, "targets.p"), "wb") as fp:
              pickle.dump(T, fp)
            tf.logging.info("DONE")

if __name__ == "__main__":
    app.run(main)