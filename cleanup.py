import os
from sigopt import Connection
import tensorflow as tf
from absl import flags, app

flags.DEFINE_string("api_token", default="",
      help="SigOpt api token")
flags.DEFINE_string("experiment_id", default="default",
      help="Experiment id")
flags.DEFINE_string("bucket_name", default="",
        help="Name of gcp bucket")
flags.DEFINE_bool("total", default=False,
        help="Total cleanup or not")

def main(_):
    
    conn = Connection(client_token=FLAGS.api_token)


    observations = conn.experiments(FLAGS.experiment_id).observations().fetch(state="failed")
    for obs in observations.iterate_pages():
        suggestion_id = str(obs.suggestion)

        model_dir = os.path.join(FLAGS.bucket_name, "models", suggestion_id)
        param_config_file = os.path.join(FLAGS.bucket_name, "param_configs", "{}.json".format(suggestion_id))

        tf.gfile.DeleteRecursively(model_dir)
        tf.gfile.Remove(param_config_file)

    conn.experiments(FLAGS.experiment_id).observations().delete(state="failed")

    if FLAGS.total:
        suggestions = conn.experiments(FLAGS.experiment_id).suggestions().fetch(state="open")
        for sugg in suggestions.iterate_pages():

            model_dir = os.path.join(FLAGS.bucket_name, "models", suggestion_id)
            param_config_file = os.path.join(FLAGS.bucket_name, "param_configs", "{}.json".format(suggestion_id))
            tf.gfile.DeleteRecursively(model_dir)
            tf.gfile.Remove(param_config_file)
        
        conn.experiments(FLAGS.experiment_id).suggestions().delete(state="open")


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    app.run(main)
