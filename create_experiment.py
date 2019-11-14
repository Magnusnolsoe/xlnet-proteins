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
      
def main(unused_args):
    del unused_args
    
    conn = Connection(client_token=FLAGS.api_token)

    experiment = conn.experiments().create(
        name=FLAGS.exp_name,
        project=FLAGS.project_id,
        metrics=[dict(name='pplx', objective='minimize')],
        parameters=[
                dict(name='mem_len', type='int', bounds=dict(min=0,max=50)),        # Multiply
                dict(name='perm_size', type='int', bounds=dict(min=1,max=2)),       # Function
                dict(name='n_layer', type='int', bounds=dict(min=1,max=6)),
                dict(name='d_model', type='int', bounds=dict(min=5,max=10)),        # Multiply
                dict(name='d_embed', type='int', bounds=dict(min=5,max=10)),        # Multiply
                dict(name='n_head', type='int', bounds=dict(min=1,max=4)),
                dict(name='d_head', type='int', bounds=dict(min=1,max=6)),          # Multiply
                dict(name='d_inner', type='int', bounds=dict(min=6,max=11)),        # Multiply
                dict(name='seq_len', type='categorical', categorical_values=['32', '64', '128', '256', '512']),
                dict(name='learning_rate', type='int', bounds=dict(min=4, max=10)), # Multiply
                dict(name='decay_method', type='categorical', categorical_values=['poly', 'cos']),
                dict(name='dropout', type='int', bounds=dict(min=0,max=5)),         # Multiply
                dict(name='dropatt', type='int', bounds=dict(min=0,max=5)),         # Multiply
                dict(name='warmup_steps', type='int', bounds=dict(min=0,max=10)),   # Multiply
                dict(name='weight_decay', type='int', bounds=dict(min=-9, max=-4))  # Mutliply
        ],
        observation_budget=FLAGS.budget,
        parallel_bandwidth=FLAGS.num_workers,
    )

    logging.info("Experiment ID: " + str(experiment.id))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)