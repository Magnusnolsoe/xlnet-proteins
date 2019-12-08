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

# Pretrain model parameters
flags.DEFINE_string("model_config_path", default="",
        help="Pretrained model config path")
flags.DEFINE_string("init_checkpoint", default="",
        help="Pretrained model checkpoint path")

# Unprocessed data dir
flags.DEFINE_string("data_dir", default="",
        help="Unprocessed data dir")

# Internal Configurations
NUM_HOSTS = 1
NUM_CORES = 8
EPOCHS = 50
FAIL_THRESHOLD = 3
ITERATIONS = 1000
SEQ_LEN = 32

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
    model_dir_basename = os.path.join("finetuning-models", dirname)
    _dir = os.path.join(FLAGS.bucket_name, model_dir_basename)
    if tf.gfile.Exists(_dir):
        tf.gfile.DeleteRecursively(_dir)
    tf.gfile.MakeDirs(_dir)

    return model_dir_basename

def generate_data_output_dir(dirname):
    output_dir_basename = os.path.join("finetuning-data", dirname)
    _dir = os.path.join(FLAGS.bucket_name, output_dir_basename)
    if tf.gfile.Exists(_dir):
        tf.gfile.DeleteRecursively(_dir)
    tf.gfile.MakeDirs(_dir)

    return output_dir_basename

def generate_param_config(dirname, suggestion_id, params, model_dir_total_path, output_data_dir):

    log_info = {"id": suggestion_id}

    tpu_zone = TPU_ZONES[FLAGS.tpu_name]

    dropout = params['dropout']/10
    dropatt = params['dropatt']/10
    if params['weight_decay'] < -8:
        weight_decay = 0
    else:
        weight_decay = pow(10, params['weight_decay'])
    warmup_steps = params['warmup_steps']*100
    lr_rate = params['learning_rate']
    lr_layer_decay_rate = params['lr_layer_decay_rate'] / 10
    batch_size = int(params['batch_size'])
    d_method = params['decay_method']

    configs = {"model_config_path": FLAGS.model_config_path , "dropout": dropout, "dropatt": dropatt, "clamp_len": None,
            "summary_type": "last", "use_summ_proj": None, "use_bfloat16": True, "init": "normal",
            "init_std": None, "init_range": None, "overwrite_data": True, "init_checkpoint": FLAGS.init_checkpoint,
            "output_dir": output_data_dir, "model_dir": model_dir_total_path, "data_dir": FLAGS.data_dir, "use_tpu": True, "num_hosts": NUM_HOSTS,
            "num_core_per_host": NUM_CORES, "tpu_job_name": None, "tpu": FLAGS.tpu_name, "tpu_zone": tpu_zone,
            "gcp_project": FLAGS.gcp_project, "master": None, "iterations": ITERATIONS, "do_train": True, "train_steps": None,
            "warmup_steps": warmup_steps, "learning_rate": lr_rate, "lr_layer_decay_rate": lr_layer_decay_rate,
            "min_lr_ratio": None, "clip": None, "max_save": None, "save_steps": None, "train_batch_size": batch_size,
            "weight_decay": weight_decay, "adam_epsilon": None, "decay_method": d_method, "do_eval": False,
            "do_predict": False, "predict_threshold": None, "eval_split": "test", "eval_batch_size": batch_size,
            "predict_batch_size": batch_size, "predict_dir": None, "eval_all_ckpt": False, "predict_ckpt": None,
            "task_name": "subloc", "max_seq_length": SEQ_LEN, "shuffle_buffer": None, "num_passes": None,
            "cls_scope": None, "is_regression": False, "python": "python3", "epochs": EPOCHS, "run_id": suggestion_id,
            "bucket_uri": FLAGS.bucket_name}

    path = os.path.join(FLAGS.bucket_name, "param_configs_finetuning", "{}.json".format(suggestion_id))
    with tf.gfile.Open(path, 'w') as fp:
        json.dump(configs, fp)
    
    return path


def start_tpu(config_path):

    with tf.gfile.Open(config_path, 'r') as config_file:
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
            'cls_scope', 'is_regression', 'run_id', 'bucket_uri'
        ]

        args = ""
        for key in param_keys:
            if params[key] is not None:
                args += "--{}={} ".format(key, params[key])
        
        python = params["python"]
        assert python is not None

        os.system(python + " train_classifier.py " + args)


def run_worker(unused_args):
    del unused_args

    conn = Connection(client_token=FLAGS.api_token)

    experiment = conn.experiments(FLAGS.experiment_id).fetch()

    worker_dir = os.path.join(FLAGS.bucket_name, "workers", str(experiment.id), FLAGS.tpu_name)
    tf.gfile.MakeDirs(worker_dir)
    worker_state_path = os.path.join(worker_dir, "status.json")


    fail_count = 0
    while experiment.progress.observation_count < experiment.observation_budget:

        suggestion = conn.experiments(experiment.id).suggestions().create(
            metadata=dict(
                host_name=FLAGS.gcp_project,
                tpu_name=FLAGS.tpu_name,
            )
        )

        # create model_dir and param config file
        model_dir_basename = generate_model_dir(suggestion.id)
        model_dir_total_path = os.path.join(FLAGS.bucket_name, model_dir_basename)
        output_data_dir_base_name = generate_data_output_dir(suggestion.id)
        output_data_dir_total_path = os.path.join(FLAGS.bucket_name, output_data_dir_base_name)
        config_path = generate_param_config(
            model_dir_basename,
            suggestion.id,
            suggestion.assignments,
            model_dir_total_path,
            output_data_dir_total_path
        )

        if start_tpu(config_path): # Only enters if failed
            observation = conn.experiments(experiment.id).observations().create(
                failed = True,
                suggestion=suggestion.id,
                metadata=dict(
                    host_name = FLAGS.gcp_project,
                    tpu_name = FLAGS.tpu_name
                )
            )
            fail_count += 1
            if fail_count >= FAIL_THRESHOLD: # Stop worker if failed FAIL_THRESHOLD or more times
                with tf.gfile.Open(worker_state_path, 'w') as f:
                    json.dump({"state": 'FAILED'}, f)
                break
            continue

        result_path = os.path.join(FLAGS.bucket_name, 'finetuning-results', "{}.json".format(suggestion.id))
        assert tf.gfile.Exists(result_path)
        with tf.gfile.Open(result_path, 'r') as result_file:
            results = json.load(result_file) # read results from suggestion generated by train_tpu.py

            values = [{'name': 'loss', 'value': float(results['loss'])}, {'name': 'acc', 'value': float(results['acc'])}]

            # Report an Observation
            observation = conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                values=values,
                metadata=dict(
                    avg_train_time=results['avg_train_time'], 
                    avg_eval_time=results['avg_eval_time'], 
                    stopped_early=results['stopped_early'], 
                    last_errors=results['last_errors'], 
                    slope=results['slope'], 
                    epoch=results['epoch'],
                    host_name = FLAGS.gcp_project,
                    tpu_name = FLAGS.tpu_name,
                )
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