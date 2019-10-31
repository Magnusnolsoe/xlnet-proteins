from absl import flags, app
import numpy as np  
import os

# GCP parameters
flags.DEFINE_string("gcp_project", default="",
      help="Name of gpc project")
flags.DEFINE_string("bucket_name", default="",
      help="Name of gcp bucket")
flags.DEFINE_string("experiment_id", default="",
      help="SigOpt experiment ID")
flags.DEFINE_string("tpu_sel", default="",
      help="TPU selection")


TPU_NAMES = {
      'V3': [],
      'V2': [],
      'PreEmpt': []
}

def spin_up_worker(api_token, experiment_id, seq_len):
      args = '--api_token={} --experiment_id={} --seq_len={} --bucket_name={} --gcp_project={}'.format(api_token, experiment_id, seq_len, FLAGS.bucket_name, FLAGS.gcp_project)
      os.system("python3 tpu_worker.py " + args) # TODO: Use another non-blocking call!

def master(unused_args):
      del unused_args

      tpus = TPU_NAMES[FLAGS.tpu_sel]


if __name__ == "__main__":
      FLAGS = flags.FLAGS
      app.run(master)