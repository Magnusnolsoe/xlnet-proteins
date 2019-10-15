# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import math

tf_training_loss_ph = tf.placeholder(tf.float32,shape=None, name='training-loss')
tf_training_pplx_ph = tf.placeholder(tf.float32,shape=None, name='training_ppl')

tf_valid_loss_ph = tf.placeholder(tf.float32,shape=None, name='validation-loss')
tf_valid_pplx_ph = tf.placeholder(tf.float32,shape=None, name='validation_ppl')

tf_test_loss_ph = tf.placeholder(tf.float32,shape=None, name='test-loss')
tf_test_acc_ph = tf.placeholder(tf.float32,shape=None, name='test-accuracy')
tf_test_pplx_ph = tf.placeholder(tf.float32,shape=None, name='test_ppl')

def tensorboard_setup(logTrain=True, logValid=True):
    
    if logTrain and logValid:
        with tf.name_scope('training'):
            tf_training_loss_summary = tf.summary.scalar('training_loss', tf_training_loss_ph)
            tf_training_ppl_summary = tf.summary.scalar('training_pplx', tf_training_pplx_ph)
            
        training_performance_summaries = tf.summary.merge([tf_training_loss_summary, tf_training_ppl_summary])
        
        with tf.name_scope('validation'):
            tf_valid_loss_summary = tf.summary.scalar('validation_loss', tf_valid_loss_ph)
            tf_valid_ppl_summary = tf.summary.scalar('validation_pplx', tf_valid_pplx_ph)
            
        valid_performance_summaries = tf.summary.merge([tf_valid_loss_summary, tf_valid_ppl_summary])
        
        return training_performance_summaries, valid_performance_summaries
    
    if logValid:
        with tf.name_scope('validation'):
            tf_valid_loss_summary = tf.summary.scalar('validation_loss', tf_valid_loss_ph)
            tf_valid_ppl_summary = tf.summary.scalar('validation_pplx', tf_valid_pplx_ph)
            
        return tf.summary.merge([tf_valid_loss_summary, tf_valid_ppl_summary])
        
    if logTrain:
        with tf.name_scope('training'):
            tf_training_loss_summary = tf.summary.scalar('training_loss', tf_training_loss_ph)
            tf_training_ppl_summary = tf.summary.scalar('training_pplx', tf_training_pplx_ph)
            
        return tf.summary.merge([tf_training_loss_summary, tf_training_ppl_summary])
    
def tensorboard_setup_test():
    with tf.name_scope('test'):
        tf_test_loss_summary = tf.summary.scalar('test_loss', tf_test_loss_ph)
        tf_test_acc_summary = tf.summary.scalar('test_acc', tf_test_acc_ph)
        tf_test_ppl_summary = tf.summary.scalar('test_pplx', tf_test_pplx_ph)
            
    return tf.summary.merge([tf_test_loss_summary, tf_test_acc_summary, tf_test_ppl_summary])

def create_writers(sess, logTrain=True, logValid=True, logging_dir='logging'):
    
        i = len(os.listdir(logging_dir)) + 1
        logging_dir = os.path.join(logging_dir, str(i))
        while os.path.exists(logging_dir):
            i += 1
            logging_dir = os.path.join(logging_dir, str(i))
        os.mkdir(logging_dir)
        
        total_train_log_dir = os.path.join(logging_dir, "train")
        total_valid_log_dir = os.path.join(logging_dir, "valid")
        
        if logTrain and logValid:
            train_summary_writer = tf.summary.FileWriter(total_train_log_dir, sess.graph)
            valid_summary_writer = tf.summary.FileWriter(total_valid_log_dir, sess.graph)
            
            return train_summary_writer, valid_summary_writer
        
        if logTrain:
            train_summary_writer = tf.summary.FileWriter(total_train_log_dir, sess.graph)
            
            return train_summary_writer
            
        if logValid:
            valid_summary_writer = tf.summary.FileWriter(total_valid_log_dir, sess.graph)
            
            return valid_summary_writer
            
def create_test_writer(sess, logging_dir='logging'):

        i = len(os.listdir(logging_dir))
        logging_dir = os.path.join(logging_dir, str(i))


        total_test_log_dir = os.path.join(logging_dir, "test")

        test_summary_writer = tf.summary.FileWriter(total_test_log_dir, sess.graph)
            
        return test_summary_writer


def run_train(sess, training_performance_summaries, curr_loss):
    return sess.run(training_performance_summaries, feed_dict={tf_training_loss_ph:curr_loss,
                                                                tf_training_pplx_ph:math.exp(curr_loss)})
            
def run_valid(sess, valid_performance_summaries, val_loss, v_pplx):
    return sess.run(valid_performance_summaries, feed_dict={tf_valid_loss_ph:val_loss,
                                                            tf_valid_pplx_ph:v_pplx})    

def run_test(sess, test_performance_summaries, test_loss, test_acc, t_pplx):
    return sess.run(test_performance_summaries, feed_dict={tf_test_loss_ph:test_loss, 
                                                           tf_test_acc_ph:test_acc,
                                                           tf_test_pplx_ph:t_pplx})