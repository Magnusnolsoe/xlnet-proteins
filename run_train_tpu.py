import os
import json
from absl import flags, app

flags.DEFINE_string('config', default=None,
      help='Parameter config file')
FLAGS = flags.FLAGS


def main(_):
    
    with open(FLAGS.config, 'r') as config_file:
        params = json.load(config_file)
        
        # train_batch_size should be equal to the bsz_per_host used in preprocessing
        param_keys = ["master", "tpu", "gcp_project", "tpu_zone", "use_tpu",
                        "num_hosts", "num_core_per_host", "track_mean",
                        "run_id", "num_passes", "record_info_dir", "model_dir",
                        "init_checkpoint", "logDir", "learning_rate", "clip",
                        "min_lr_ratio", "warmup_steps", "adam_epsilon",
                        "decay_method", "weight_decay", "batch_size",
                        "train_steps", "iterations", "save_steps", "max_save",
                        "seq_len", "reuse_len", "perm_size", "bi_data",
                        "mask_alpha", "mask_beta", "num_predict", "n_token",
                        "mem_len", "same_length", "clamp_len", "n_layer",
                        "d_model", "d_embed", "n_head", "d_head", "d_inner",
                        "dropout", "dropatt", "untie_r", "summary_type", 
                        "ff_activation", "use_bfloat16", "init", "init_std",
                        "init_range", "bucket_uri"]
        
        args = ""
        for key in param_keys:
            if params[key] is not None:
                args += "--{}={} ".format(key, params[key])

        
        python = params['python']
        assert python is not None

        os.system(python + " train_tpu.py " + args)


if __name__ == "__main__":
    app.run(main)        