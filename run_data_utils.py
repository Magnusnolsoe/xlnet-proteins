# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:55:17 2019

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
        
        '''
        python = params['python'] # python cmd to be used
        use_tpu = params['use_tpu'] # whether or not to use tpu
        use_eop = params['use_eop'] # whether or not to use EOP token at the end of each protein
        batch_size_per_host = params['bsz_per_host'] # 
        number_core_per_host = params['num_core_per_host']
        input_file = params['input_file'] # name of the input file (including file extension) 
        data_dir = params['data_dir'] # path of input data directory
        save_dir = params['save_dir'] # path of the save data directory
        seq_len = params['seq_len'] # sequence length
        reuse_len = params['reuse_len'] # reuse length
        bi_data = params['bi_data'] # whether or not to create bidirectional data
        num_predict = params['num_predict'] # number of tokens to predict
        mask_alpha = params['mask_alpha'] # mask alpha
        mask_beta = params['mask_beta'] # mask beta
        '''
        
        param_keys = ["use_tpu", "use_eop", "bsz_per_host", "input_file",
                      "data_dir", "save_dir", "seq_len", "reuse_len",
                      "bi_data", "num_predict", "mask_alpha", "mask_beta"]
        
        args = ""
        for key in param_keys:
            if params[key] is not None:
                args += "--{}={} ".format(key, params[key])
        
        python = params['python'] # python cmd to be used
        assert python is not None

        os.system(python +" data_utils.py " + args)

if __name__ == "__main__":
        app.run(main)