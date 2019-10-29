from sigopt import Connection
from absl import flags, app
import numpy as np  

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

# Experiment parameters
flags.DEFINE_integer("seq_len", default=0,
      help="Seq len of dataset")

MAX_WARMUP_STEPS = 5000

def spin_up_worker(experiment_id):
    
    # TODO: Create folders, config files and start worker
    pass

def master(_):

    conn = Connection(client_token=FLAGS.api_token)

    experiment = conn.experiments().create(
        name=FLAGS.name,
        project=FLAGS.project_id,
        metrics=[dict(name='pplx', objective='minimize')],
        parameters=[
            dict(name='mem_len', type='int', bounds=dict(min=0,max=500)),
            dict(name='perm_size', type='int', bounds=dict(min=1,max=FLAGS.seq_len/2)),
            dict(name='n_layer', type='int', bounds=dict(min=1,max=24)),
            dict(name='d_model', type='int', bounds=dict(min=1,max=1024)),
            dict(name='d_embed', type='int', bounds=dict(min=1,max=1024)),
            dict(name='n_head', type='int', bounds=dict(min=1,max=16)),
            dict(name='d_head', type='int', bounds=dict(min=1,max=64)),
            dict(name='d_inner', type='int', bounds=dict(min=1,max=4096)),
            dict(name='batch_size', type='categorical', categorical_values=[64, 32, 16, 8]),
            dict(name='learning_rate', type='double', bounds=dict(min=1e-6, max=1e-1)),
            dict(name='dropout', type='double', bounds=dict(min=0,max=1)),
            dict(name='dropatt', type='double', bounds=dict(min=0,max=1)),
            dict(name='warmup_steps', type='int', bounds=dict(min=0,max=MAX_WARMUP_STEPS)),
            dict(name='weight_decay', type='categorical', categorical_values=[1.e-08, 1.e-07, 1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01])
        ],
        metadata=dict(
            'avg_training_time'=0,
            'avg_evaluation_time'=0
        ),
        observation_budget=FLAGS.budget,
        parallel_bandwidth=FLAGS.num_workers,
    )

    for _ in range(FLAGS.num_workers):
        spin_up_worker(experiment.id)

if __name__ == "__main__":

    
    FLAGS = flags.FLAGS

    app.run(master)