import os
import tensorflow as tf

def get_logdir(_dir):

    if not tf.gfile.Exists(_dir):
        tf.gfile.MakeDirs(_dir)

    i = len(tf.io.gfile.listdir(_dir)) + 1
    logging_dir_n = os.path.join(_dir, str(i))
    while tf.gfile.Exists(logging_dir_n):
        i += 1
        logging_dir_n = os.path.join(_dir, str(i))
    logging_dir = logging_dir_n
    tf.gfile.MakeDirs(logging_dir)
    return logging_dir