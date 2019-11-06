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

def main(_):
    
    conn = Connection(client_token=FLAGS.api_token)


    suggestions = conn.experiments(FLAGS.experiment_id).suggestions().fetch(state="failed")
    for suggestion in suggestions:
        model_dir = os.path.join(FLAGS.bucket_name, "models", suggestion.id)
        param_config_file = os.path.join(FLAGS.bucket_name, "param_configs", "{}.json".format(suggestion.id))

        tf.gfile.DeleteRecursively(model_dir)
        tf.gfile.Remove(param_config_file)

    conn.experiments(FLAGS.experiment_id).suggestions().delete(state="failed")


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    app.run(main)
