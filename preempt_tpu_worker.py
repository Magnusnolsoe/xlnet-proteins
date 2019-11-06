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
flags.DEFINE_string("tpu_name", default="",
        help="TPU name")
flags.DEFINE_string("seq_len", default="",
        help="Sequence length")

# Internal Configurations
NUM_HOSTS = 1
NUM_CORES = 8
EPOCHS = 50
ITERATIONS = 50000

TPU_ZONES = {
    'instance-1': "us-central1-a",
    'instance-2': "us-central1-a",
    'instance-3': "us-central1-a",
    'instance-4': "us-central1-a",
    'instance-5': "us-central1-a",
    'v2-1': "us-central1-f",
    'v2-2': "us-central1-f",
    'v2-3': "us-central1-f",
    'v2-4': "us-central1-f",
    'v2-5': "us-central1-f",
    'preempt-1': "us-central1-f",
    'preempt-2': "us-central1-f",
    'preempt-3': "us-central1-f",
    'preempt-4': "us-central1-f",
    'preempt-5': "us-central1-f",
    'preempt-6': "us-central1-f",
    'preempt-7': "us-central1-f",
    'preempt-8': "us-central1-f",
    'preempt-9': "us-central1-f",
    'preempt-10': "us-central1-f"
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
    mem_len = params['mem_len']*10
    perm_size = params['perm_size']
    n_layer = params['n_layer']
    d_model = pow(2,params['d_model'])
    d_embed = pow(2,params['d_embed'])
    n_head = params['n_head']
    d_head = pow(2,params['d_head'])
    d_inner = pow(2,params['d_inner'])
    batch_size = int(params['batch_size'])
    lr_rate = params['learning_rate']
    dropout = params['dropout']/10
    dropatt = params['dropatt']/10
    warmup_steps = params['warmup_steps']*100
    if params['weight_decay'] == 0:
        weight_decay = 0
    else:
        weight_decay = pow(10, params['weight_decay'])

    seq_len = int(FLAGS.seq_len)
    reuse_len = seq_len // 2
    if seq_len == 512:
        n_pred = 85
    else:
        n_pred = int(round(0.15*seq_len))
    record_info_dir = get_record_info_dir(reuse_len, n_pred, batch_size)
    tpu_zone = TPU_ZONES[FLAGS.tpu_name]

    configs = {"master": None, "tpu": FLAGS.tpu_name, "gcp_project": FLAGS.gcp_project,
                 "tpu_zone": tpu_zone, "use_tpu": True, "num_hosts": NUM_HOSTS,
                 "num_core_per_host": NUM_CORES, "track_mean": True,
                 "run_id": suggestion_id, "num_passes": None, "record_info_dir": record_info_dir, "model_dir": dirname,
                 "init_checkpoint": None, "logDir": 'logging', "learning_rate": lr_rate, "clip": None,
                 "min_lr_ratio": None, "warmup_steps": warmup_steps, "adam_epsilon": None,
                 "decay_method": 'poly', "weight_decay": weight_decay, "batch_size": batch_size,
                 "train_steps": None, "iterations": ITERATIONS, "save_steps": None, "max_save": None,
                 "seq_len": seq_len, "reuse_len": reuse_len, "perm_size": reuse_len, 
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

    experiment = conn.experiments(FLAGS.experiment_id).fetch()

    worker_dir = os.path.join(FLAGS.bucket_name, "workers", str(experiment.id), FLAGS.tpu_name)
    tf.gfile.MakeDirs(worker_dir)
    worker_state_path = os.path.join(worker_dir, "status.json")


    didIFail = False
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

            suggestion_id = str(suggestion.id)

            tf.logging.info("Failed with suggestion: {}, deleting and starting again".format(suggestion_id))

            model_dir = os.path.join(FLAGS.bucket_name, "models", suggestion_id)
            param_config_file = os.path.join(FLAGS.bucket_name, "param_configs", "{}.json".format(suggestion_id))

            tf.gfile.DeleteRecursively(model_dir)
            tf.gfile.Remove(param_config_file)

            conn.experiments(experiment.id).suggestions(suggestion_id).delete()

            didIFail = True

            break

        result_path = os.path.join(FLAGS.bucket_name, 'results', "{}.json".format(suggestion.id))
        assert tf.gfile.Exists(result_path)
        with tf.gfile.Open(result_path, 'r') as result_file:
            results = json.load(result_file) # read results from suggestion generated by train_tpu.py

            # Report an Observation
            observation = conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                value=float(results['pplx']),
                metadata=dict(avg_train_time=results['avg_train_time'], avg_eval_time=results['avg_eval_time'], stopped_early=results['stopped_early'], last_errors=results['last_errors'], slope=results['slope'], epoch=results['epoch'])
            )

            # Update the experiment object
            experiment = conn.experiments(experiment.id).fetch()

        tf.gfile.DeleteRecursively(model_dir_total_path)
    
    if not didIFail:
        with tf.gfile.Open(worker_state_path, 'w') as f:
            json.dump({"state": 'DONE'}, f)
    else:
        with tf.gfile.Open(worker_state_path, 'w') as f:
            json.dump({"state": 'FAILED'}, f)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(run_worker)