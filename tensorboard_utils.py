# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import math

tf_training_loss_ph = tf.placeholder(tf.float32,shape=None, name='training-loss')
tf_training_acc_ph = tf.placeholder(tf.float32,shape=None, name='training-accuracy')
tf_training_pplx_ph = tf.placeholder(tf.float32,shape=None, name='training_ppl')

tf_valid_loss_ph = tf.placeholder(tf.float32,shape=None, name='validation-loss')
tf_valid_acc_ph = tf.placeholder(tf.float32,shape=None, name='validation-accuracy')
tf_valid_pplx_ph = tf.placeholder(tf.float32,shape=None, name='validation_ppl')


def tensorboard_setup(logTrain=True, logValid=True):
    
    if logTrain and logValid:
        with tf.name_scope('training'):
            tf_training_loss_summary = tf.summary.scalar('training_loss', tf_training_loss_ph)
            tf_training_acc_summary = tf.summary.scalar('training_accuracy', tf_training_acc_ph)
            tf_training_ppl_summary = tf.summary.scalar('training_pplx', tf_training_pplx_ph)
            
        training_performance_summaries = tf.summary.merge([tf_training_loss_summary, tf_training_acc_summary, tf_training_ppl_summary])
        
        with tf.name_scope('validation'):
            tf_valid_loss_summary = tf.summary.scalar('validation_loss', tf_valid_loss_ph)
            tf_valid_acc_summary = tf.summary.scalar('validation_acc', tf_valid_acc_ph)
            tf_valid_ppl_summary = tf.summary.scalar('validation_pplx', tf_valid_pplx_ph)
            
        valid_performance_summaries = tf.summary.merge([tf_valid_loss_summary, tf_valid_acc_summary, tf_valid_ppl_summary])
        
        return training_performance_summaries, valid_performance_summaries
    
    if logValid:
        with tf.name_scope('validation'):
            tf_valid_loss_summary = tf.summary.scalar('validation_loss', tf_valid_loss_ph)
            tf_valid_acc_summary = tf.summary.scalar('validation_acc', tf_valid_acc_ph)
            tf_valid_ppl_summary = tf.summary.scalar('validation_pplx', tf_valid_pplx_ph)
            
        return tf.summary.merge([tf_valid_loss_summary, tf_valid_acc_summary, tf_valid_ppl_summary])
        
    if logTrain:
        with tf.name_scope('training'):
            tf_training_loss_summary = tf.summary.scalar('training_loss', tf_training_loss_ph)
            tf_training_acc_summary = tf.summary.scalar('training_accuracy', tf_training_acc_ph)
            tf_training_ppl_summary = tf.summary.scalar('training_pplx', tf_training_pplx_ph)
            
        return tf.summary.merge([tf_training_loss_summary, tf_training_acc_summary, tf_training_ppl_summary])
    
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
            
        
def run_train(sess, training_performance_summaries, curr_loss, curr_acc):
    return sess.run(training_performance_summaries, feed_dict={tf_training_loss_ph:curr_loss,
                                                                tf_training_acc_ph:curr_acc,
                                                                tf_training_pplx_ph:math.exp(curr_loss)})
            
def run_valid(sess, valid_performance_summaries, val_loss, val_acc, v_pplx):
    return sess.run(valid_performance_summaries, feed_dict={tf_valid_loss_ph:val_loss, 
                                                            tf_valid_acc_ph:val_acc, 
                                                            tf_valid_pplx_ph:v_pplx})
            