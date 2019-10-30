import socket
import os
import json
import tensorflow as tf

from sigopt import Connection
from absl import flags, app
from os_utils import get_logdir
from data_utils import VOCAB_SIZE

# SigOpt parameters
flags.DEFINE_string("api_token", default="",
        help="SigOpt api token")
flags.DEFINE_string("experiment_id", default="",
        help="SigOpt experiment ID")

# Google Cloud Platform parameters
flags.DEFINE_string("gcp_project", default="",
        help="Name of gpc project")
flags.DEFINE_string("bucket_name", default="",
        help="Name of gcp bucket")

# TPU parameters
flags.DEFINE_string("seq_len", default="",
        help="Sequence length")

# Constants
NUM_HOSTS = 5
NUM_CORES = 8
EPOCHS = 10
# TODO: Fill <FILL_OUT> out!!!
TPUS = {
    '128': 'instance-2',
    '256': '<FILL_OUT>',
    '512': '<FILL_OUT>'
}
TPU_ZONES = {
    '128': 'us-central1-a',
    '256': '<FILL_OUT>',
    '512': '<FILL_OUT>'
}

def generate_model_dir(dirname):
    model_dir_basename = os.path.join("models", dirname)
    _dir = os.path.join(FLAGS.bucket_name, model_dir_basename)
    if tf.gfile.Exists(_dir):
        tf.gfile.DeleteRecursively(_dir)
    tf.gfile.MakeDirs(_dir)

    return model_dir_basename

def get_record_info_dir(reuse_len, n_pred, bsz):

    basename = "seq_len{}-reuse_len{}-n_pred{}-bsz{}".format(FLAGS.seq_len, reuse_len, n_pred, bsz)

    return os.path.join("proc_data", basename)

def generate_param_config(dirname, suggestion_id, params):

    log_info = {"id": suggestion_id}

    # Suggestions from SigOpt
    mem_len = params['mem_len']
    perm_size = params['perm_size']
    n_layer = params['n_layer']
    d_model = params['d_model']
    d_embed = params['d_embed']
    n_head = params['n_head']
    d_head = params['d_head']
    d_inner = params['d_inner']
    batch_size = int(params['batch_size'])
    lr_rate = params['learning_rate']
    dropout = params['dropout']
    dropatt = params['dropatt']
    warmup_steps = params['warmup_steps']
    weight_decay = float(params['weight_decay'])

    # TPU parameters
    tpu = TPUS[FLAGS.seq_len]
    zone = TPU_ZONES[FLAGS.seq_len]

    seq_len = int(FLAGS.seq_len)
    reuse_len = seq_len // 2
    n_pred = int(round(0.15*seq_len))
    record_info_dir = get_record_info_dir(reuse_len, n_pred, batch_size)

    configs = {"master": None, "tpu": tpu, "gcp_project": FLAGS.gcp_project,
                 "tpu_zone": zone, "use_tpu": True, "num_hosts": NUM_HOSTS,
                 "num_core_per_host": NUM_CORES, "track_mean": True,
                 "run_id": suggestion_id, "num_passes": None, "record_info_dir": record_info_dir, "model_dir": dirname,
                 "init_checkpoint": None, "logDir": 'logging', "learning_rate": lr_rate, "clip": None,
                 "min_lr_ratio": None, "warmup_steps": warmup_steps, "adam_epsilon": None,
                 "decay_method": 'poly', "weight_decay": weight_decay, "batch_size": batch_size,
                 "train_steps": None, "iterations": None, "save_steps": None, "max_save": None,
                 "seq_len": FLAGS.seq_len, "reuse_len": reuse_len, "perm_size": reuse_len, 
                 "bi_data": False, "mask_alpha": 6, "mask_beta": 1, "num_predict": n_pred, "n_token": VOCAB_SIZE,
                 "mem_len": mem_len, "same_length": None, "clamp_len": None, "n_layer": n_layer,
                 "d_model": d_model, "d_embed": d_embed, "n_head": n_head, "d_head": d_head, "d_inner": d_inner,
                 "dropout": dropout, "dropatt": dropatt, "untie_r": None, "summary_type": 'last', 
                 "ff_activation": 'relu', "use_bfloat16": True, "init": 'normal', "init_std": None,
                 "init_range": None, "bucket_uri": FLAGS.bucket_name, "epochs": EPOCHS, "python": "python3"}

    path = os.path.join(FLAGS.bucket_name, "param_configs", "{}.json".format(suggestion_id))
    with tf.gfile.Open(path, 'w') as fp:
        json.dump(configs, fp)
    
    return path


def start_tpu(config_path):

    with tf.gfile.Open(config_path, 'r') as config_file:
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
                    "init_range", "bucket_uri", "epochs"]
        
    args = ""
    for key in param_keys:
        if params[key] is not None:
            args += "--{}={} ".format(key, params[key])

    python = params['python']
    # returns 0 if failed, and 1 if succeeded
    return os.system(python + " train_tpu.py " + args)


def run_worker(unused_args):
    del unused_args

    conn = Connection(client_token=FLAGS.api_token)
    hostname = socket.gethostname()

    experiment = conn.experiments(FLAGS.experiment_id).fetch()

    worker_dir = os.path.join(FLAGS.bucket_name, "workers", str(experiment.id))
    tf.gfile.MakeDirs(worker_dir)
    worker_state_path = os.path.join(worker_dir, "status.json")
    with tf.gfile.Open(worker_state_path, 'w') as fp:
        json.dump({"state": 'ALIVE'}, fp)


    fail_count = 0
    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()

        # create model_dir and param config file
        model_dir_basename = generate_model_dir(suggestion.id)
        model_dir_total_path = os.path.join(FLAGS.bucket_name, model_dir_basename)
        config_path = generate_param_config(
            model_dir_basename,
            suggestion.id,
            suggestion.assignments
        )


        if start_tpu(config_path): # Only enters if failed
            fail_log = suggestion.assignments
            with tf.gfile.Open(os.path.join(worker_dir, "{}.json".format(suggestion.id))) as f:
                json.dump(fail_log, f)
            conn.experiments(experiment.id).suggestions(suggestion.id).delete()
            fail_count += 1
            if fail_count >= 3: # Stop worker if failed 3 or more times
                with tf.gfile.Open(worker_state_path, 'w') as f:
                    json.dump({"state": 'FAILED'}, f)
                break
        
        result_path = os.path.join(model_dir_total_path, 'results.json')
        assert tf.gfile.Exists(result_path)
        with tf.gfile.Open(result_path, 'r') as result_file:
            results = json.load(result_file) # read results from suggestion generated by train_tpu.py

            # Report an Observation
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                value=results['pplx'],
                metadata=dict(avg_train_time=results['avg_train_time'], avg_eval_time=results['avg_eval_time'])
            )

            # Update the experiment object
            experiment = conn.experiments(experiment.id).fetch()

        tf.gfile.DeleteRecursively(model_dir_total_path)
        fail_count = 0
    
    if fail_count < 3:
        with tf.gfile.Open(worker_state_path, 'w') as f:
                json.dump({"state": 'DONE'}, f)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(run_worker)