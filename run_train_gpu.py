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
        
        python = params['python']
        
        # GPU config
        num_hosts = params['num_hosts']
        num_core_per_host = params['num_core_per_host']
        use_tpu = params['use_tpu']
        
        # Experiment (data/checkpoint/directory) config
        num_passes = params['num_passes']
        record_info_dir = params['record_info_dir']
        model_dir = params['model_dir'] 
        init_checkpoint = params['init_checkpoint']
        
        # Optimization config
        lr = params['learning_rate']
        clip = params['clip']
        min_lr_ratio = params['min_lr_ratio']
        warmup_steps = params['warmup_steps']
        epsilon = params['adam_epsilon']
        dec_meth = params['decay_method']
        weight_dec = params['weight_decay']
        
        # Training config
        train_bsz = params['bsz_per_host']
        train_steps = params['train_steps']
        iters = params['iterations']
        save_steps = params['save_steps']
        perm_size = params['perm_size']
        n_token = params['n_token']
        
        # Model config
        mem_len = params['mem_len']
        same_len = params['same_length']
        clamp_len = params['clamp_length']
        n_layer = params['n_layer']
        d_model = params['d_model']
        d_embed = params['d_embed']
        n_head = params['n_head']
        d_inner = params['d_inner']
        drp_rate = params['dropout']
        dropatt = params['dropatt']
        untie_r = params['untie_r']
        sum_type = params['summary_type']
        ff_act = params['ff_activation']
        use_bfloat = params['use_bfloat16']
        
        # Parameter initialization
        init = params['init']
        init_std = params['init_std']
        init_range = params['init_range']
        
        args = ('--num_hosts={} '
                '--num_core_per_host={} '
                '--use_tpu={} '
                '--num_passes={} '
                '--record_info_dir={} '
                '--model_dir={} '
                '--init_checkpoint={} '
                '--learning_rate={} '
                '--clip={} '
                '--min_lr_ratio={} '
                '--warmup_steps={} '
                '--adam_epsilon={} '
                '--decay_method={} '
                '--weight_decay={} '
                '--train_batch_size={} '
                '--train_steps={} '
                '--iterations={} '
                '--save_steps={} '
                '--perm_size={} '
                '--n_token={} '
                '--mem_len={} '
                '--same_length={} '
                '--clamp_len={} '
                '--n_layer={} '
                '--d_model={} '
                '--d_embed={} '
                '--n_head={} '
                '--d_inner={} '
                '--dropout={} '
                '--dropatt={} '
                '--untie_r={} '
                '--summary_type={} '
                '--ff_activation={} '
                '--use_bfloat16={} '
                '--init={} '
                '--init_std={} '
                '--init_range={} '
                ).format(num_hosts, num_core_per_host, use_tpu,
                num_passes, record_info_dir, model_dir, init_checkpoint,
                lr, clip, min_lr_ratio, warmup_steps, epsilon, dec_meth,
                weight_dec, train_bsz, train_steps, iters, save_steps,
                perm_size, n_token, mem_len, same_len, clamp_len, n_layer,
                d_model, d_embed, n_head, d_inner, drp_rate, dropatt, untie_r,
                sum_type, ff_act, use_bfloat, init, init_std, init_range
                )
        
        os.system(python + " run_train_gpu.py " + args)


if __name__ == "__main__":
    app.run(main)        