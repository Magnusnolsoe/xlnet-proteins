import json
import os
import tensorflow as tf

def create_examples(lines, set_type):
    """Creates examples for the training and test sets."""
    d = dict()
    total = 0
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)

      line = line.split(":")

      seq = line[0].strip()
      label = line[1].strip()

      if label in d:
        d[label] += 1
      else:
        d[label] = 1

      total += 1
    return d, total

def read_txt(input_file):
    """Reads a /n separated value file"""
    with tf.gfile.Open(input_file, "r") as f:
      lines = []
      for line in f:
        if len(line) == 0: continue
        lines.append(line)
      return lines

def get_train_examples(data_dir):
    return create_examples(
      read_txt(os.path.join(data_dir, train_file)), "train")

def get_test_examples(data_dir):
    return create_examples(
        read_txt(os.path.join(data_dir, test_file)), "test")

train_file = "train.txt"
test_file = "test.txt"

train_data, train_total = get_train_examples("data")
test_data, test_total = get_test_examples("data")

print("Printing training: *************************: Total examples: ", str(train_total))
for x in train_data:
    print(repr(x),":",train_data[x], "percentage: ", str(int(train_data[x])/train_total *100))
print("Printing testing: **************************: Total examples: ", str(test_total))
for x in test_data:
    print(repr(x),":",test_data[x], "percentage: ", str(int(test_data[x])/test_total * 100))