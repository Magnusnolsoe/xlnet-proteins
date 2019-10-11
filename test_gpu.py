import tensorflow as tf
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import data_utils

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("model_name", default=None,
      help="Estimator model_name.")

# GPU config
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")

# Testing config
flags.DEFINE_integer("test_batch_size", default=16,
      help="Size of test batch.")

# Data config
flags.DEFINE_integer('seq_len', default=0,
      help='Sequence length for testing.')
flags.DEFINE_integer('reuse_len', default=0,
      help="How many tokens to be reused in the next batch. "
      "Could be half of seq_len")
flags.DEFINE_bool("bi_data", default=True,
      help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=6,
      help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
      help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
      help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer('perm_size', default=None,
  help='perm size.')
flags.DEFINE_integer("n_token", 27, help="Vocab size")


flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")

FLAGS = flags.FLAGS

def test(ps_device, model_path):

    test_input_fn, record_info_dict_test = data_utils.get_input_fn(
          info_dir=os.path.join(FLAGS.record_info_dir, "test"),
          split="test",
          bsz_per_host=FLAGS.test_batch_size,
          seq_len=FLAGS.seq_len,
          reuse_len=FLAGS.reuse_len,
          bi_data=FLAGS.bi_data,
          num_hosts=1,
          num_core_per_host=1,
          perm_size=FLAGS.perm_size,
          mask_alpha=FLAGS.mask_alpha,
          mask_beta=FLAGS.mask_beta,
          use_bfloat16=FLAGS.use_bfloat16,
          num_predict=FLAGS.num_predict)

    tf.logging.info("num of test batches {}".format(record_info_dict_test["num_batch"]))

    ##### Create input tensors / placeholders
    bsz_per_core = FLAGS.test_batch_size // FLAGS.num_core_per_host

    params = {
        "batch_size": FLAGS.test_batch_size # the whole batch
    }
    test_set = test_input_fn(params)

    t_iter = test_set.make_initializable_iterator()
    t_example = t_iter.get_next()

    if FLAGS.num_core_per_host > 1:
        # test set
        t_examples = [{} for _ in range(FLAGS.num_core_per_host)]
        for key in t_example.keys():
            vals = tf.split(t_examples[key], FLAGS.num_core_per_host, 0)
            for device_id in range(FLAGS.num_core_per_host):
                t_examples[device_id][key] = vals[device_id]
    else:
        t_example = [t_example]


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    FLAGS.n_token = data_utils.VOCAB_SIZE
    tf.logging.info("n_token {}".format(FLAGS.n_token))

    model_total_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)

    assert tf.gfile.Exists(model_total_path)

    test("/gpu:0", model_total_path)

if __name__ == "__main__":
    tf.app.run()