import os


def get_logdir(_dir):
    i = len(os.listdir(_dir)) + 1
    logging_dir_n = os.path.join(_dir, str(i))
    while os.path.exists(logging_dir_n):
        i += 1
        logging_dir_n = os.path.join(_dir, str(i))
    logging_dir = logging_dir_n
    os.mkdir(logging_dir)
    return logging_dir