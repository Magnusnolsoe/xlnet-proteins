from sigopt import Connection
from absl import flags, app

def spin_up_worker(api_token, experiment_id):
    # TODO: Create folders, config files and start worker
    pass

def master(_):

    conn = Connection(client_token=api_token)

    experiment = conn.experiments().create(
        name=FLAGS.name,
        project=FLAGS.project,
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
        spin_up_worker(api_token, experiment.id)

if __name__ == "__main__":

    
    FLAGS = flags.FLAGS

    app.run(master)