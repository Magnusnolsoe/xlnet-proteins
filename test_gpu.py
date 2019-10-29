from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorboard_utils as tb
import os
import numpy as np
import math

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import data_utils
import model_utils
import function_builder

from gpu_utils import assign_to_gpu

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")

# GPU config
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
      help="Whether to use TPUs for training.")

# Testing config
flags.DEFINE_integer("test_batch_size", default=16,
      help="Size of test batch.")

# Model config
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
      help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")

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

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# Logging config
flags.DEFINE_string("tb_logging_dir", default="logging",
                    help="The directory to save the logs for Tensorboard.")

FLAGS = flags.FLAGS

def get_model_fn():
  def model_fn(features, labels, mems):
    #### Get loss from inputs
    total_loss, new_mems = function_builder.get_loss(
        FLAGS, features, labels, mems, False)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # GPU
    #assert is_training
    return total_loss, new_mems

  return model_fn


def single_core_graph(features, mems):
  model_fn = get_model_fn()

  model_ret = model_fn(
      features=features,
      labels=None,
      mems=mems)

  return model_ret


def create_mems_tf(bsz_per_core):
  mems = [tf.placeholder(dtype=tf.float32,
                         shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model])
          for layer in range(FLAGS.n_layer)]

  return mems


def initialize_mems_np(bsz_per_core):
  mems_np = [np.zeros(shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model],
                      dtype=np.float32)
             for layer in range(FLAGS.n_layer)]

  return mems_np

def test(ps_device):

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
        t_examples = [t_example]

    ##### Create computational graph
    v_tower_mems, v_tower_losses, v_tower_new_mems = [], [], []

    for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), \
            tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            # The mems for each tower is a dictionary
            v_mems_i = {}
            if FLAGS.mem_len:
                v_mems_i["mems"] = create_mems_tf(bsz_per_core)
            
            v_loss_i, v_new_mems_i = single_core_graph(
                features=t_examples[i],
                mems=v_mems_i)
            
            v_tower_mems.append(v_mems_i)
            v_tower_losses.append(v_loss_i)
            v_tower_new_mems.append(v_new_mems_i)

    ## average losses and gradients across towers
    if len(v_tower_losses) > 1:
      v_loss = tf.add_n(v_tower_losses) / len(v_tower_losses)
    else:
      v_loss = v_tower_losses[0]

    gpu_options = tf.GPUOptions(allow_growth=True)

    model_utils.init_from_checkpoint(FLAGS, global_vars=True)

    # Create performance summaries for Tensorboard logging
    test_performance_summaries = tb.tensorboard_setup_test()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # Create writers for Tensorboard logging
        test_summary_writer = tb.create_test_writer(sess, logging_dir=FLAGS.tb_logging_dir)

        # initialize mems
        v_tower_mems_np = []
        for i in range(FLAGS.num_core_per_host):
            v_mems_i_np = {}
        for key in v_tower_mems[i].keys():
            v_mems_i_np[key] = initialize_mems_np(bsz_per_core)
            v_tower_mems_np.append(v_mems_i_np)
        
        v_fetches = [v_loss, v_tower_new_mems]
        
        sess.run(t_iter.initializer)
        v_total_loss = 0.
        v_steps = 0

        try:
            while True:
                v_feed_dict = {}
                for i in range(FLAGS.num_core_per_host):
                    for key in v_tower_mems_np[i].keys():
                        for m, m_np in zip(v_tower_mems[i][key], v_tower_mems_np[i][key]):
                            v_feed_dict[m] = m_np
                    
                v_fetched = sess.run(v_fetches, feed_dict=v_feed_dict)
                v_loss_np, v_tower_mems_np = v_fetched[:]
                v_total_loss += v_loss_np
                v_steps += 1
                print(v_steps)
            
        except tf.errors.OutOfRangeError:
            test_loss = v_total_loss/v_steps
            t_pplx = math.exp(test_loss)
            tf.logging.info("Test: loss {:.2f} | pplx {:>7.2f}".format(
                            test_loss,  t_pplx))
            
            summ_test = tb.run_test(sess, test_performance_summaries, test_loss, t_pplx)
            test_summary_writer.add_summary(summ_test, 1)

def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    FLAGS.n_token = data_utils.VOCAB_SIZE
    tf.logging.info("n_token {}".format(FLAGS.n_token))

    assert FLAGS.init_checkpoint is not None

    test("/gpu:0")

if __name__ == "__main__":
    tf.app.run()