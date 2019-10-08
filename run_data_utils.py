# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:55:17 2019

@author: s144471
"""

import os


python = 'python' # python cmd to be used
use_tpu = False # whether or not to use tpu
use_eop = True # whether or not to use EOP token at the end of each protein
batch_size_per_host = 32 # 
number_core_per_host = 1
input_file = '' # name of the input file (including file extension) 
data_dir = '' # path of input data directory
save_dir = '' # path of the save data directory
seq_len = 512 # sequence length
reuse_len = 256 # reuse length
bi_data = True # whether or not to create bidirectional data
num_predict = 85 # number of tokens to predict
mask_alpha = 6 # mask alpha
mask_beta = 1 # mask beta

args = "--use_tpu={} --bsz_per_host={} --num_core_per_host={} --input_file={} --save_dir={} --data_dir={} --use_eop={} --seq_len={} --reuse_len={} --bi_data={} --num_predict={} --mask_alpha={} --mask_beta={}".format(use_tpu, batch_size_per_host, number_core_per_host, input_file, save_dir, data_dir, use_eop, seq_len, reuse_len, bi_data, num_predict, mask_alpha, mask_beta)

os.system(python +" data_utils.py " + args)