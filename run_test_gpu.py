# -*- coding: utf-8 -*-
"""
Created on Tue Oct  11 09:14:05 2019

@author: s144440
"""

import os
import json
from absl import flags, app

flags.DEFINE_string('config', default=None,
      help='Parameter config file')
FLAGS = flags.FLAGS

def main(_):
    
    with open(FLAGS.config, 'r') as config_file:
        params = json.load(config_file)
        
        param_keys = ["record_info_dir", "num_core_per_host", "bsz_per_host",
                      "seq_len", "reuse_len", "bi_data", "mask_alpha", "mask_beta",
                      "num_predict", "perm_size", "use_bfloat16", "n_token",
                      "model_dir", "model_name"]
        
        args = ""
        for key in param_keys:
            if params[key] is not None:
                if key == "bsz_per_host":
                    args += "--test_batch_size={} ".format(params[key])
                else:
                    args += "--{}={} ".format(key, params[key])

        
        python = params['python']
        assert python is not None

        os.system(python + " test_gpu.py " + args)

if __name__ == "__main__":
    app.run(main)        