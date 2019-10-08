# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:57:30 2019

@author: s144471
"""

import os
import json
from absl import flags, app

flags.DEFINE_string('param_config', default=None,
      help='Parameter config file')
FLAGS = flags.FLAGS


def main(_):
    
    with open(FLAGS.param_config, 'r') as config_file:
        params = json.load(config_file)
        
        
        param_keys = ["num_hosts", "num_core_per_host",
                      "use_tpu", "num_passes", "record_info_dir",
                      "model_dir", "init_checkpoint", "learning_rate",
                      "clip", "min_lr_ratio", "warmup_steps", "adam_epsilon",
                      "decay_method", "weight_decay", "bsz_per_host",
                      "train_steps", "iterations", "save_steps", "perm_size",
                      "n_token", "mem_len", "same_length", "clamp_len",
                      "n_layer", "d_model", "d_embed", "n_head", "d_inner",
                      "dropout", "dropatt", "untie_r", "summary_type",
                      "ff_activation", "use_bfloat16", "init", "init_std",
                      "init_range"]
        
        args = ""
        for key in param_keys:
            if params[key] is not None:
                if key == "bsz_per_host":
                    args += "--train_batch_size={} ".format(params[key])
                else:
                    args += "--{}={} ".format(key, params[key])

        
        python = params['python']
        assert python is not None

        os.system(python + " train_gpu.py " + args)


if __name__ == "__main__":
    app.run(main)        