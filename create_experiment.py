from sigopt import Connection
from absl import flags, app
from tensorflow import logging

# SigOpt parameters
flags.DEFINE_string("exp_name", default="default",
      help="Experiment name")
flags.DEFINE_string("project_id", default="",
      help="Project id")
flags.DEFINE_integer("budget", default=1,
      help="Experiment observation budget")
flags.DEFINE_integer("num_workers", default=1,
      help="SigOpt parallel bandwidth")
flags.DEFINE_string("api_token", default="",
      help="SigOpt api token")
flags.DEFINE_integer("seq_len", default=0,
      help="Sequence length")

def main(unused_args):
    del unused_args
    
    conn = Connection(client_token=FLAGS.api_token)

    batches = {
        '128': ['64', '32', '16', '8'],
        '256': ['32', '16', '8'],
        '512': ['16', '8']
    }
    experiment = conn.experiments().create(
        name=FLAGS.exp_name,
        project=FLAGS.project_id,
        metrics=[dict(name='pplx', objective='minimize')],
        parameters=[
                dict(name='mem_len', type='int', bounds=dict(min=0,max=500)),
                dict(name='perm_size', type='int', bounds=dict(min=1,max=FLAGS.seq_len//2)),
                dict(name='n_layer', type='int', bounds=dict(min=1,max=24)),
                dict(name='d_model', type='int', bounds=dict(min=1,max=1024)),
                dict(name='d_embed', type='int', bounds=dict(min=1,max=1024)),
                dict(name='n_head', type='int', bounds=dict(min=1,max=16)),
                dict(name='d_head', type='int', bounds=dict(min=1,max=64)),
                dict(name='d_inner', type='int', bounds=dict(min=1,max=4096)),
                dict(name='batch_size', type='categorical', categorical_values=batches[seq_len]),
                dict(name='learning_rate', type='double', bounds=dict(min=1e-6, max=1e-2)),
                dict(name='dropout', type='double', bounds=dict(min=0,max=0.8)),
                dict(name='dropatt', type='double', bounds=dict(min=0,max=0.8)),
                dict(name='warmup_steps', type='int', bounds=dict(min=0,max=1000)),
                dict(name='weight_decay', type='categorical', categorical_values=list(map(str, [0, 1.e-08, 1.e-07, 1.e-06, 1.e-05, 1.e-04]))) # TODO: figure out values to use since we only have 10 catagorical values!
        ],
        observation_budget=FLAGS.budget,
        parallel_bandwidth=FLAGS.num_workers,
    )

    logging.info("Experiment ID: " + str(experiment.id))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)