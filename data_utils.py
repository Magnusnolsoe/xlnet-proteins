# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:58:47 2019

@author: s144471
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys
import json

LOOKUP_TABLE = {'M': 0, 'R': 1, 'W': 2, 'L': 3, 'D': 4,
                'K': 5, 'F': 6, 'G': 7, 'E': 8, 'S': 9,
                'V': 10, 'A': 11, 'H': 12, 'T': 13,'P': 14,
                'I': 15, 'N': 16, 'C': 17, 'Y': 18, 'Q': 19,
                'X': 20, 'B': 21, 'U': 22, 'Z': 23}
special_symbols = {
    "<eop>"  : 24,
}

VOCAB_SIZE = len(LOOKUP_TABLE) + len(special_symbols)
EOP_ID = special_symbols["<eop>"]


def parse_files_to_dataset(parser, file_names, split, num_batch, num_hosts,
                           host_id, num_core_per_host, bsz_per_core):
    
    
    #assert split == "train"
    dataset = tf.data.Dataset.from_tensor_slices(file_names)
    
    if len(file_names) > 1:
        dataset = dataset.shuffle(len(file_names))
    
    dataset = tf.data.TFRecordDataset(dataset)
    
    # (zihang): since we are doing online preprocessing, the parsed result of
    # the same input at each time will be different. Thus, cache processed data
    # is not helpful. It will use a lot of memory and lead to contrainer OOM.
    # So, change to cache non-parsed raw data instead.
    dataset = dataset.cache().map(parser)
    if split == "train":
        dataset = dataset.repeat()
    dataset = dataset.batch(bsz_per_core, drop_remainder=True)
    dataset = dataset.prefetch(num_core_per_host * bsz_per_core)
    
    return dataset

def _convert_example(example, use_bfloat16):
      """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
      for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
          val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
          val = tf.cast(val, tf.int32)
        if use_bfloat16 and val.dtype == tf.float32:
          val = tf.cast(val, tf.bfloat16)
    
        example[key] = val

def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.
    Args:
        inputs: int64 Tensor in shape [seq_len], input ids.
        targets: int64 Tensor in shape [seq_len], target ids.
        is_masked: bool Tensor in shape [seq_len]. True means being selected
            for partial prediction.
        perm_size: the length of longest permutation. Could be set to be reuse_len.
        Should not be larger than reuse_len or there will be data leaks.
        seq_len: int, sequence length.
    """
    
    # Generate permutation indices
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])
    
    
    non_mask_tokens = tf.logical_not(is_masked)
    
    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    smallest_index = -tf.ones([seq_len], dtype=tf.int64)
    rev_index = tf.where(non_mask_tokens, smallest_index, index)

    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_mask = tf.cast(is_masked, tf.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    self_rev_index = tf.where(is_masked, rev_index, rev_index + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = tf.logical_and(
            self_rev_index[:, None] <= rev_index[None, :],
            is_masked)
    perm_mask = tf.cast(perm_mask, tf.float32)

    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = tf.concat([inputs[0: 1], targets[: -1]],
                          axis=0)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = target_mask

    return perm_mask, new_targets, target_mask, inputs_k, inputs_q


def get_dataset(params, num_hosts, num_core_per_host, split, file_names,
                num_batch, seq_len, reuse_len, perm_size, mask_alpha,
                mask_beta, use_bfloat16=False, num_predict=None):
    
    bsz_per_core = params["batch_size"]
    if num_hosts > 1:
        host_id = params["context"].current_host
    else:
        host_id = 0
        
    def parser(record):
        """function used to parse tfrecord."""
        
        record_spec = {
            "input": tf.FixedLenFeature([seq_len], tf.int64),
            "target": tf.FixedLenFeature([seq_len], tf.int64),
            "seg_id": tf.FixedLenFeature([seq_len], tf.int64),
            "label": tf.FixedLenFeature([1], tf.int64),
            "is_masked": tf.FixedLenFeature([seq_len], tf.int64),
        }
        
        # retrieve serialized example
        example = tf.parse_single_example(
            serialized=record,
            features=record_spec
        )
        
        inputs = example.pop("input")
        target = example.pop("target")
        is_masked = tf.cast(example.pop("is_masked"), tf.bool)
        
        non_reuse_len = seq_len - reuse_len
        assert perm_size <= reuse_len and perm_size <= non_reuse_len
        
        
        perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
            inputs[:reuse_len],
            target[:reuse_len],
            is_masked[:reuse_len],
            perm_size,
            reuse_len)

        perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
            inputs[reuse_len:],
            target[reuse_len:],
            is_masked[reuse_len:],
            perm_size,
            non_reuse_len)

        perm_mask_0 = tf.concat([perm_mask_0, tf.ones([reuse_len, non_reuse_len])],
                            axis=1)
        perm_mask_1 = tf.concat([tf.zeros([non_reuse_len, reuse_len]), perm_mask_1],
                            axis=1)
        perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)
        target = tf.concat([target_0, target_1], axis=0)
        target_mask = tf.concat([target_mask_0, target_mask_1], axis=0)
        input_k = tf.concat([input_k_0, input_k_1], axis=0)
        input_q = tf.concat([input_q_0, input_q_1], axis=0)

        if num_predict is not None:
            indices = tf.range(seq_len, dtype=tf.int64)
            bool_target_mask = tf.cast(target_mask, tf.bool)
            indices = tf.boolean_mask(indices, bool_target_mask)
        
            actual_num_predict = tf.shape(indices)[0]

            ##### target_mapping
            target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
            example["target_mapping"] = tf.reshape(target_mapping,
                                                 [num_predict, seq_len])
    
            ##### target
            target = tf.boolean_mask(target, bool_target_mask)
            example["target"] = tf.reshape(target, [num_predict])
    
            ##### target mask
            target_mask = tf.ones([actual_num_predict], dtype=tf.float32)
            example["target_mask"] = tf.reshape(target_mask, [num_predict])
        else:
          example["target"] = tf.reshape(target, [seq_len])
          example["target_mask"] = tf.reshape(target_mask, [seq_len])

        # reshape back to fixed shape
        example["perm_mask"] = tf.reshape(perm_mask, [seq_len, seq_len])
        example["input_k"] = tf.reshape(input_k, [seq_len])
        example["input_q"] = tf.reshape(input_q, [seq_len])

        _convert_example(example, use_bfloat16)

        for k, v in example.items():
            tf.logging.info("%s: %s", k, v)

        return example
        
    dataset = parse_files_to_dataset(
      parser=parser,
      file_names=file_names,
      split=split,
      num_batch=num_batch,
      num_hosts=num_hosts,
      host_id=host_id,
      num_core_per_host=num_core_per_host,
      bsz_per_core=bsz_per_core)
    
    return dataset

    
def get_input_fn(
    info_dir,
    split,
    bsz_per_host,
    seq_len,
    reuse_len,
    bi_data,
    num_hosts=1,
    num_core_per_host=1,
    perm_size=None,
    mask_alpha=None,
    mask_beta=None,
    use_bfloat16=False,
    num_predict=None,
    use_tpu=False,
	bucket_uri=None):
    
    basename = format_filename("record-info", bsz_per_host, seq_len,
                                bi_data, "json", mask_alpha=mask_alpha,
                                mask_beta=mask_beta, reuse_len=reuse_len,
                                fixed_num_predict=num_predict)

    if bucket_uri is not None:
        record_info_path = os.path.join(os.path.join(bucket_uri, info_dir), basename)
    else:
        record_info_path = os.path.join(info_dir, basename)

    assert tf.io.gfile.exists(record_info_path)

    record_info = {"num_batch": 0, "filenames": []}

    tf.logging.info("Using the following record info dir: %s", info_dir)
    tf.logging.info("Record info path: %s", record_info_path)
    
    with tf.gfile.Open(record_info_path, "r") as fp:
        info = json.load(fp)
        
        record_info["num_batch"] += info["num_batch"]
        if bucket_uri is not None:
            record_info["filenames"] = [os.path.join(bucket_uri, f) for f in info["filenames"]]
        else:
            record_info["filenames"] = info["filenames"]

    tf.logging.info("Total number of batches: %d",
                    record_info["num_batch"])
    tf.logging.info("Total number of files: %d",
                    len(record_info["filenames"]))
    tf.logging.info(record_info["filenames"])
    
    
    def input_fn(params):
        """docs."""
        assert params["batch_size"] * num_core_per_host == bsz_per_host
        
        dataset = get_dataset(
                params=params,
                num_hosts=num_hosts,
                num_core_per_host=num_core_per_host,
                split=split,
                file_names=record_info["filenames"],
                num_batch=record_info["num_batch"],
                seq_len=seq_len,
                reuse_len=reuse_len,
                perm_size=perm_size,
                mask_alpha=mask_alpha,
                mask_beta=mask_beta,
                use_bfloat16=use_bfloat16,
                num_predict=num_predict)
        
        return dataset
    
    return input_fn, record_info

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def preprocess_protein(protein_seq):
    return protein_seq.strip().split(' ')

def encode_ids(protein_seq):
    encoded = []
    for amino in protein_seq:
        if amino in LOOKUP_TABLE:
            encoded.append(LOOKUP_TABLE[amino])
        else:
            encoded.append(special_symbols[amino])
    return encoded

def batchify(data, bsz_per_host, prot_ids=None):
    num_step = len(data) // bsz_per_host
    data = data[:bsz_per_host * num_step]
    data = data.reshape(bsz_per_host, num_step)
    if prot_ids is not None:
        prot_ids = prot_ids[:bsz_per_host * num_step]
        prot_ids = prot_ids.reshape(bsz_per_host, num_step)

    if prot_ids is not None:
        return data, prot_ids
    return data

def format_filename(prefix, bsz_per_host, seq_len, bi_data, suffix,
                    mask_alpha=5, mask_beta=1, reuse_len=None,
                    fixed_num_predict=None):
    """docs."""
    
    if reuse_len is None:
        reuse_len_str = ""
    else:
        reuse_len_str = "reuse-{}.".format(reuse_len)
    if bi_data:
        bi_data_str = "bi"
    else:
        bi_data_str = "uni"
    if fixed_num_predict is not None:
        fnp_str = "fnp-{}.".format(fixed_num_predict)
    else:
        fnp_str = ""
        
    file_name = "{}.bsz-{}.seqlen-{}.{}{}.alpha-{}.beta-{}.{}{}".format(
      prefix, bsz_per_host, seq_len, reuse_len_str, bi_data_str,
      mask_alpha, mask_beta, fnp_str, suffix)
    
    return file_name

def _sample_mask(seg, alpha, beta, reverse=False, max_gram=5, goal_num_predict=None):
    """Sample `goal_num_predict` tokens for partial prediction.
    About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""
    
    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)
    num_predict = 0
    
    ngrams = np.arange(1, max_gram+1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True)
    
    if reverse:
        seg = np.flip(seg, 0)
        
    cur_len = 0
    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break
        
        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)
        ctx_size = (n * alpha) // beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx
        
        # Find start position of a complete token
        beg = cur_len + l_ctx
        if beg >= seg_len:
            break
        
        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + n
        if end > seg_len:
            break
        
        # Update
        mask[beg:end] = True
        num_predict += end - beg
        
        cur_len = end + r_ctx
    
    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1
    
    if reverse:
        mask = np.flip(mask, 0)
    
    return mask
        

def create_tfrecords(save_dir, basename, data, bsz_per_host, seq_len,
                     bi_data):
    """DOC_STRING"""
    
    data, prot_ids = data
    
    num_core = FLAGS.num_core_per_host
    bsz_per_core = bsz_per_host // num_core

    if bi_data:
        assert bsz_per_host % (2 * FLAGS.num_core_per_host) == 0
        fwd_data, fwd_prot_ids = batchify(data, bsz_per_host // 2, prot_ids)
        
        fwd_data = fwd_data.reshape(num_core, 1, bsz_per_core // 2, -1)
        fwd_prot_ids = fwd_prot_ids.reshape(num_core, 1, bsz_per_core // 2, -1)
        
        bwd_data = fwd_data[:, :, :, ::-1]
        bwd_prot_ids = fwd_prot_ids[:, :, :, ::-1]
        
        data = np.concatenate(
            [fwd_data, bwd_data], 1).reshape(bsz_per_host, -1)
        prot_ids = np.concatenate(
            [fwd_prot_ids, bwd_prot_ids], 1).reshape(bsz_per_host, -1)
    else:
        data, prot_ids = batchify(data, bsz_per_host, prot_ids)
    
    
    tf.logging.info("Raw data shape %s.", data.shape)
    
    file_name = format_filename(
        prefix=basename,
        bsz_per_host=bsz_per_host,
        seq_len=seq_len,
        bi_data=bi_data,
        suffix="tfrecords",
        mask_alpha=FLAGS.mask_alpha,
        mask_beta=FLAGS.mask_beta,
        reuse_len=FLAGS.reuse_len,
        fixed_num_predict=FLAGS.num_predict
    )
    save_path = os.path.join(save_dir, file_name)
    tf.logging.info("Start writing %s.", save_path)
    
    num_batch = 0
    reuse_len = FLAGS.reuse_len
    
    # [sep] x 2 + [cls]
    assert reuse_len < seq_len
    
    
    data_len = data.shape[1]
    
    with tf.python_io.TFRecordWriter(save_path) as record_writer:
        i = 0
        while i + seq_len <= data_len:
            if num_batch % 500 == 0:
                tf.logging.info("Processing batch {}".format(num_batch))
                
            all_ok = True
            features = []
            for idx in range(bsz_per_host):
                inp = data[idx, i: i+seq_len]
                tgt = data[idx, i+1: i+seq_len+1]
                
                # sample ngram spans to predict
                reverse = bi_data and (idx // (bsz_per_core // 2)) % 2 == 1
                mask_0 = _sample_mask(inp, FLAGS.mask_alpha, FLAGS.mask_beta,
                                      reverse=reverse, goal_num_predict=FLAGS.num_predict)
                
                # Concatenate data
                seg_id = [0] * seq_len
                
                
                is_masked = mask_0
                if FLAGS.num_predict is not None:
                    assert np.sum(is_masked) == FLAGS.num_predict
                
                label = 1
                feature = {
                  "input": _int64_feature(inp),
                  "is_masked": _int64_feature(is_masked),
                  "target": _int64_feature(tgt),
                  "seg_id": _int64_feature(seg_id),
                  "label": _int64_feature([label]),
                }
                features.append(feature)
            
            if all_ok:
                assert len(features) == bsz_per_host
                for feature in features:
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    record_writer.write(example.SerializeToString())
                num_batch += 1
            else:
                break
            
            i += reuse_len
            
    tf.logging.info("Done writing %s. Num of batches: %d", save_path, num_batch)
    
    return (save_path, num_batch)

def _create_data(data_path):
    """TODO: Maybe read glob of files and concat!"""
    
    input_data, prot_ids = [], []
    prot_id = True
    for line_cnt, line in enumerate(tf.io.gfile.GFile(data_path)):
        if line_cnt % 10000 == 0:
            tf.logging.info("Loading line {}".format(line_cnt))
        
        if not line.strip():
            continue
        else:
            cur_prot = preprocess_protein(line)
            cur_prot = encode_ids(cur_prot)
            if FLAGS.use_eop:
                cur_prot.append(EOP_ID)
            
        input_data.extend(cur_prot)
        prot_ids.extend([prot_id]*len(cur_prot))
        
        prot_id = not prot_id
    
    
    input_data = np.array(input_data, dtype=np.int64)
    prot_ids = np.array(prot_ids, dtype=np.bool)
    
    file_name, cur_num_batch = create_tfrecords(
          save_dir=FLAGS.save_dir,
          basename="{}".format(os.path.basename(data_path).split('.')[0]),
          data=(input_data, prot_ids),
          bsz_per_host=FLAGS.bsz_per_host,
          seq_len=FLAGS.seq_len,
          bi_data=FLAGS.bi_data
    )
    
    record_info = {
                "filenames": [file_name],
                "num_batch": cur_num_batch
            }
    
    return record_info
        
def create_data(_):
    """DOC_STRING"""
    
    if not FLAGS.use_tpu:
        FLAGS.num_core_per_host = 1  # forced to be one

    # Validate FLAGS
    assert FLAGS.bsz_per_host % FLAGS.num_core_per_host == 0
    
    train_path = os.path.join(FLAGS.data_dir, FLAGS.train_filename)
    valid_path = os.path.join(FLAGS.data_dir, FLAGS.valid_filename)
    test_path = os.path.join(FLAGS.data_dir, FLAGS.test_filename)
    if not tf.io.gfile.exists(train_path):
        tf.logging.error("File {} does not exist".format(train_path))
        return
    
    if not tf.io.gfile.exists(valid_path):
        tf.logging.error("File {} does not exist".format(valid_path))
        return
    
    if not tf.io.gfile.exists(test_path):
        tf.logging.error("File {} does not exist".format(test_path))
        return

    # Make workdirs

    # train save dirs
    train_save_path = os.path.join(FLAGS.save_dir, "train")
    if not tf.io.gfile.exists(train_save_path):
        tf.gfile.MakeDirs(train_save_path)

    # valid save dirs
    valid_save_path = os.path.join(FLAGS.save_dir, "valid")
    if not tf.io.gfile.exists(valid_save_path):
        tf.gfile.MakeDirs(valid_save_path)

    # test save dirs
    test_save_path = os.path.join(FLAGS.save_dir, "test")
    if not tf.io.gfile.exists(test_save_path):
        tf.gfile.MakeDirs(test_save_path)
    

    if FLAGS.make_train_set:
        tf.logging.info("Processing training data \"{}\"".format(train_path))
        record_info = _create_data(train_path)
        record_name = format_filename(
            prefix="record-info",
            bsz_per_host=FLAGS.bsz_per_host,
            seq_len=FLAGS.seq_len,
            mask_alpha=FLAGS.mask_alpha,
            mask_beta=FLAGS.mask_beta,
            reuse_len=FLAGS.reuse_len,
            bi_data=FLAGS.bi_data,
            suffix="json",
            fixed_num_predict=FLAGS.num_predict)
        record_info_path = os.path.join(train_save_path, record_name)
        
        with tf.gfile.Open(record_info_path, "w") as fp:
            json.dump(record_info, fp)
    
    if FLAGS.make_valid_set:
        tf.logging.info("Processing validation data \"{}\"".format(valid_path))
        record_info = _create_data(valid_path)
        record_name = format_filename(
            prefix="record-info",
            bsz_per_host=FLAGS.bsz_per_host,
            seq_len=FLAGS.seq_len,
            mask_alpha=FLAGS.mask_alpha,
            mask_beta=FLAGS.mask_beta,
            reuse_len=FLAGS.reuse_len,
            bi_data=FLAGS.bi_data,
            suffix="json",
            fixed_num_predict=FLAGS.num_predict)
        record_info_path = os.path.join(valid_save_path, record_name)
        
        with tf.gfile.Open(record_info_path, "w") as fp:
            json.dump(record_info, fp)

    if FLAGS.make_test_set:
        tf.logging.info("Processing validation data \"{}\"".format(test_path))
        record_info = _create_data(test_path)
        record_name = format_filename(
            prefix="record-info",
            bsz_per_host=FLAGS.bsz_per_host,
            seq_len=FLAGS.seq_len,
            mask_alpha=FLAGS.mask_alpha,
            mask_beta=FLAGS.mask_beta,
            reuse_len=FLAGS.reuse_len,
            bi_data=FLAGS.bi_data,
            suffix="json",
            fixed_num_predict=FLAGS.num_predict)
        record_info_path = os.path.join(test_save_path, record_name)
    
        with tf.gfile.Open(record_info_path, "w") as fp:
            json.dump(record_info, fp)

if __name__ == "__main__":
    flags = tf.app.flags
    
    # DEFINE SOME FLAGS HERE:
    
    # TPU parameters:
    flags.DEFINE_bool("use_tpu", True, help="whether to use TPUs")
    flags.DEFINE_integer("bsz_per_host", 32, help="batch size per host.")
    flags.DEFINE_integer("num_core_per_host", 8, help="num TPU cores per host.")
    
    # Experiment config
    flags.DEFINE_string("train_filename", default="", help="Filename of training set.")
    flags.DEFINE_string("valid_filename", default="", help="Filename of validation set.")
    flags.DEFINE_string("test_filename", default="", help="Filename of test set.")
    flags.DEFINE_string("save_dir", default="proc_data",
                      help="Directory for saving the processed data.")
    flags.DEFINE_string("data_dir", default="data",
                      help="Directory with raw input data.")
    flags.DEFINE_bool("use_eop", True,
                    help="whether to append EOP at the end of a protein.")
    flags.DEFINE_bool("make_train_set", False,
                    help="whether to build the train dataset.")
    flags.DEFINE_bool("make_valid_set", False,
                    help="whether to build the valid dataset.")
    flags.DEFINE_bool("make_test_set", False,
                    help="whether to build the test dataset.")
    


    # Data config
    flags.DEFINE_integer("seq_len", 512,
                       help="Sequence length.")
    flags.DEFINE_integer("reuse_len", 256,
                       help="Number of token that can be reused as memory. "
                       "Could be half of `seq_len`.")
    flags.DEFINE_bool("bi_data", True,
                    help="whether to create bidirectional data")
    flags.DEFINE_integer("num_predict", default=85,
                       help="Num of tokens to predict.")
    flags.DEFINE_integer("mask_alpha", default=6,
                       help="How many tokens to form a group.")
    flags.DEFINE_integer("mask_beta", default=1,
                       help="How many tokens to mask within each group.")
    FLAGS = flags.FLAGS
    
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(create_data)