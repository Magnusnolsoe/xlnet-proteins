import pickle
import os
import torch
import torch.optim as optim

from tqdm import tqdm
from absl import app, flags
from dataloader import DataLoader
from lstm_model import SeqClassifier

flags.DEFINE_string("train_data", default="",
      help="Path to training data")
flags.DEFINE_string("test_data", default="",
      help="Path to test data")
flags.DEFINE_integer("batch_size", default=32,
      help="Batch size")
flags.DEFINE_bool("shuffle", default=True,
      help="Whether or not to shuffle data")
flags.DEFINE_bool("use_gpu", default=True,
      help="Whether or not to use GPU when training")

# Model parameters
flags.DEFINE_integer("d_embed", default=1024,
      help="Dimension of XLNet output embeddings")
flags.DEFINE_integer("d_project", default=256,
      help="Projection dimension of model")
flags.DEFINE_integer("d_rnn", default=256,
      help="Dimension of hidden states in the LSTM")
flags.DEFINE_integer("n_layers", default=2,
      help="Layers in the LSTM")
flags.DEFINE_bool("bidirection", default=True,
      help="Use Bi-LSTM or not")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate for linear layers")
flags.DEFINE_float("rnn_dropout", default=0.0,
      help="Dropout rate for LSTM")

# Training parameters
flags.DEFINE_integer("epochs", default=100,
      help="Number of epochs to run")
flags.DEFINE_float("learning_rate", default=3e-4,
      help="Learning rate of optimizer")
flags.DEFINE_float("weight_decay", default=0.0,
      help="Weight decay")
    

FLAGS = flags.FLAGS


def main(_):
    
    train_inputs = pickle.load(open(os.path.join(FLAGS.train_data, "embeddings.p"), 'rb'))
    train_targets = pickle.load(open(os.path.join(FLAGS.train_data, "targets.p"), 'rb'))
    
    test_inputs = pickle.load(open(os.path.join(FLAGS.test_data, "embeddings.p"), 'rb'))
    test_targets = pickle.load(open(os.path.join(FLAGS.test_data, "targets.p"), 'rb'))

    train_iter = DataLoader(train_inputs, train_targets, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle, d_embed=FLAGS.d_embed)
    test_iter = DataLoader(train_inputs, train_targets, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle, d_embed=FLAGS.d_embed)

    device = torch.device("cuda" if FLAGS.use_gpu and torch.cuda.is_available() else "cpu")

    model = SeqClassifier(device, d_embed=FLAGS.d_embed, d_project=FLAGS.d_project,
                    d_rnn=FLAGS.d_rnn, n_layers=FLAGS.n_layers, bi_dir=FLAGS.bidirection,
                    dropout=FLAGS.dropout, rnn_dropout=FLAGS.rnn_dropout).to(device)

    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    train_err, test_err = [], []
    train_acc, test_acc = [], []

    for epoch in range(FLAGS.epochs):
        
        print("Epoch: {} / {}".format(epoch+1, FLAGS.epochs))

        ### TRAIN LOOP ###
        err, acc = [], []
        model.train()
        for inp, tar, seqlen in tqdm(train_iter, ascii=False, desc="Training", total=int(len(train_inputs) / FLAGS.batch_size), unit="batch"):

            inputs = inp.to(device)
            targets = tar.to(device)
            seqlens = seqlen.to(device)

            predictions = model(inputs, seqlens)

            optimizer.zero_grad()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            err.append(loss.cpu().item())
            #acc.append(accuracy.cpu().item())

        epoch_trainig_error = sum(err) / len(err)
        train_err.append(epoch_trainig_error)

        ### TEST LOOP ###
        err, acc = [], []
        model.eval()
        for inp, tar, seqlen in  tqdm(test_iter, ascii=False, desc="Evaluating", total=int(len(test_inputs) / FLAGS.batch_size), unit="batch"):
            inputs = inp.to(device)
            targets = tar.to(device)
            seqlens = seqlen.to(device)

            predictions = model(inputs, seqlens)

            loss = criterion(predictions, targets)


            err.append(loss.cpu().item())
        
        epoch_test_error = sum(err) / len(err)
        test_err.append(epoch_test_error)

        logger.info("Training error: {0:.4f} | Test error: {1:.4f}".format(epoch_trainig_error, epoch_test_error))

if __name__ == "__main__":
    app.run(main)