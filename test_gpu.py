import tensorflow as tf
import os

import data_utils

def test(ps_device):

    test_input_fn, record_info_dict_test = data_utils.get_input_fn(
          info_dir=os.path.join(FLAGS.record_info_dir, "test"),
          split="test",
          bsz_per_host=FLAGS.train_batch_size,
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

def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    FLAGS.n_token = data_utils.VOCAB_SIZE
    tf.logging.info("n_token {}".format(FLAGS.n_token))

    model_total_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)

    assert tf.gfile.Exists(model_total_path)

    test("/gpu:0")

if __name__ == "__main__":
    tf.app.run()