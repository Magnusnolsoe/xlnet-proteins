from sigopt import Connection
from absl import flags, app
import numpy as np  
import os

# SigOpt parameters
flags.DEFINE_string("name", default="",
      help="Experiment name")
flags.DEFINE_string("project_id", default="",
      help="Project id")
flags.DEFINE_integer("budget", default=0,
      help="Experiment observation budget")
flags.DEFINE_integer("num_workers", default=0,
      help="SigOpt parallel bandwidth")
flags.DEFINE_string("api_token", default="",
      help="SigOpt api token")

# GCP parameters
flags.DEFINE_string("gcp_project", default="",
        help="Name of gpc project")
flags.DEFINE_string("bucket_name", default="",
        help="Name of gcp bucket")

MAX_WARMUP_STEPS = 5000

def spin_up_worker(api_token, experiment_id, seq_len):
      args = '--api_token={} --experiment_id={} --seq_len={} --bucket_name={} --gcp_project={}'.format(api_token, experiment_id, seq_len, FLAGS.bucket_name, FLAGS.gcp_project)
      os.system("python3 tpu_worker.py " + args) # TODO: Use another non-blocking call!

def master(unused_args):
    del unused_args

    conn = Connection(client_token=FLAGS.api_token)

    seq_lens = ['128']#, '256', '512']
    batches = {
          '128': ['64', '32', '16', '8'],
          '256': ['32', '16', '8'],
          '512': ['16', '8']
    }
    budgets = {
          '128': FLAGS.budget,
          '256': FLAGS.budget,
          '512': FLAGS.budget
    }

    for seq_len in seq_lens:
      experiment = conn.experiments().create(
            name=FLAGS.name.join(seq_len),
            project=FLAGS.project_id,
            metrics=[dict(name='pplx', objective='minimize')],
            parameters=[
                  dict(name='mem_len', type='int', bounds=dict(min=0,max=500)),
                  dict(name='perm_size', type='int', bounds=dict(min=1,max=int(seq_len)//2)),
                  dict(name='n_layer', type='int', bounds=dict(min=1,max=24)),
                  dict(name='d_model', type='int', bounds=dict(min=1,max=1024)),
                  dict(name='d_embed', type='int', bounds=dict(min=1,max=1024)),
                  dict(name='n_head', type='int', bounds=dict(min=1,max=16)),
                  dict(name='d_head', type='int', bounds=dict(min=1,max=64)),
                  dict(name='d_inner', type='int', bounds=dict(min=1,max=4096)),
                  dict(name='batch_size', type='categorical', categorical_values=batches[seq_len]),
                  dict(name='learning_rate', type='double', bounds=dict(min=1e-6, max=1e-1)),
                  dict(name='dropout', type='double', bounds=dict(min=0,max=1)),
                  dict(name='dropatt', type='double', bounds=dict(min=0,max=1)),
                  dict(name='warmup_steps', type='int', bounds=dict(min=0,max=MAX_WARMUP_STEPS)),
                  dict(name='weight_decay', type='categorical', categorical_values=list(map(str, [1.e-08, 1.e-07, 1.e-06, 1.e-05, 1.e-04]))) # TODO: figure out values to use since we only have 10 catagorical values!
            ],
            observation_budget=budgets[seq_len],
            parallel_bandwidth=FLAGS.num_workers,
      )

      spin_up_worker(FLAGS.api_token, experiment.id, seq_len)

if __name__ == "__main__":

    
    FLAGS = flags.FLAGS

    app.run(master)