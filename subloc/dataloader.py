import torch
import numpy as np
import pickle

from tqdm import tqdm


class DataIterator(object):
    def __init__(self, inputs, targets, sequence_lengths,
                 batch_size=0, pad_sequences=True):
        
        self.batch_size = batch_size
        self.pad_sequences = pad_sequences
        self.inputs = inputs
        self.targets = targets
        self.sequence_lengths = sequence_lengths
        self.data_pointer = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        inputs_batch = []
        targets_batch =[]
        seq_len_batch = []
        
        if (self.data_pointer >= len(self.inputs)):
            self.data_pointer = 0
            raise StopIteration
            
        end = min(len(self.inputs), self.data_pointer+self.batch_size)
        
        inputs_batch = self.inputs[self.data_pointer:end]
        targets_batch = self.targets[self.data_pointer:end]
        seq_len_batch = self.sequence_lengths[self.data_pointer:end]
        
        
        self.data_pointer += self.batch_size
        
        if (self.pad_sequences):
            padded_inputs = self.pad_input_sequence(inputs_batch, seq_len_batch)
            padded_targets = self.pad_targets(targets_batch, seq_len_batch)
            return torch.stack(padded_inputs), torch.tensor(seq_len_batch, dtype=torch.int), torch.stack(padded_targets)
        
        return torch.stack(inputs_batch), torch.tensor(seq_len_batch, dtype=torch.int), targets_batch
    
    
    def pad_input_sequence(self, inputs, lengths):
        
        max_length = lengths[0]
        padded = []
        for x in inputs:
            seq_len = len(x)
            padded.append(torch.nn.functional.pad(x, (0, max_length-seq_len), mode='constant', value=20))
            
        return padded
    
    def pad_targets(self, targets, lengths):
    
        max_length = lengths[0]
        padded = []
        for profile, seq_len in zip(targets, lengths):
            padded.append(T(torch.nn.functional.pad(
                            T(profile), (0, max_length-seq_len), value=1)))
            
        return padded


class DataLoader(object):
	def __init__(self, path="", verbose=False):
		self.path = path
		self.inputs = []
		self.targets = []
		self.sequence_lengths = []
		self.verbose = verbose

	def run_pipeline(self, split_rate):

		self.load_data()

		X_train, X_test, y_train, y_test, seq_train, seq_test = self.split(split_rate)

		X_train_sorted, y_train_sorted, seq_train_sorted = self.sort_data(X_train, y_train, seq_train)
		X_test_sorted, y_test_sorted, seq_test_sorted = self.sort_data(X_test, y_test, seq_test)

		return (X_train_sorted, X_test_sorted), (y_train_sorted, y_test_sorted), (seq_train_sorted, seq_test_sorted)


	def load_data(self):
		with open(self.path, 'r') as file:
			for line in (tqdm(file, ascii=True, desc="Loading Data", total=get_num_lines(self.path), unit="lines") if self.verbose else file):
				inputs = line.split("\t")[0]
				outputs = line.split("\t")[1]
				inputs = inputs.split(",")
				inputs = list(map(int, inputs))
				inputs = torch.tensor(inputs, dtype=torch.long)
				outputs = outputs.split(",")
				outputs = [x.split(" ") for x in outputs]
				outputs = [list(map(float, x)) for x in outputs]
				outputs = torch.tensor(outputs, dtype=torch.float)
				self.inputs.append(inputs)
				self.sequence_lengths.append(len(inputs))
				self.targets.append(outputs)

	def sort_data(self, inputs, targets, seq_lengths):
		sorted_data = sorted(zip(seq_lengths, range(len(inputs)), inputs, targets), reverse = True)

		i = [x for _,_,x,_ in sorted_data]
		t = [x for _,_,_,x in sorted_data]            
		seq_len = [x for x,_,_,_ in sorted_data]
		return i, t, seq_len

	def split(self, split_rate=0.33):
		X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(self.inputs, self.targets, self.sequence_lengths,
																					test_size=split_rate)

		return X_train, X_test, y_train, y_test, seq_train, seq_test