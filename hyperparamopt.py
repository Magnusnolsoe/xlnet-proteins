from sigopt import Connection
from absl import flags, app

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
            dict(name='', type='', bounds=dict(min=,max=)),
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