import os
import json
from absl import flags, app
import tensorflow as tf

flags.DEFINE_string('config', default=None,
      help='Parameter config file')
FLAGS = flags.FLAGS

def main(_):

    with tf.gfile.Open(FLAGS.config, 'r') as config_file:
        params = json.load(config_file)

        param_keys = [
            'model_config_path', 'dropout', 'dropatt', 'clamp_len',
            'summary_type', 'use_summ_proj', 'use_bfloat16', 'init',
            'init_std', 'init_range', 'overwrite_data', 'init_checkpoint',
            'output_dir', 'model_dir', 'data_dir', 'use_tpu', 'num_hosts',
            'num_core_per_host', 'tpu_job_name', 'tpu', 'tpu_zone',
            'gcp_project', 'master', 'iterations', 'do_train', 'train_steps',
            'warmup_steps', 'learning_rate', 'lr_layer_decay_rate',
            'min_lr_ratio', 'clip', 'max_save', 'save_steps', 'train_batch_size',
            'weight_decay', 'adam_epsilon', 'decay_method', 'do_eval',
            'do_predict', 'predict_threshold', 'eval_split', 'eval_batch_size',
            'predict_batch_size', 'predict_dir', 'eval_all_ckpt', 'predict_ckpt',
            'task_name', 'max_seq_length', 'shuffle_buffer', 'num_passes',
            'cls_scope', 'is_regression', 'run_id', 'bucket_uri', 'epochs'
        ]

        args = ""
        for key in param_keys:
            if params[key] is not None:
                args += "--{}={} ".format(key, params[key])
        
        python = params["python"]
        assert python is not None

        os.system(python + " train_classifier.py " + args)

if __name__ == "__main__":
    app.run(main)