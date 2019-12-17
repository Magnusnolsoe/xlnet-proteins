import torch
import numpy as np
import random

class DataLoader(object):
    def __init__(self, inputs, targets, batch_size=0, shuffle=True, d_embed=1024):
        
        self.batch_size = batch_size
        self.data_pointer = 0
        self.d_embed = d_embed
        if shuffle:
            self.inputs, self.targets = self.shuffle(inputs, targets)
        else:
            self.inputs = inputs
            self.targets = targets

    def shuffle(self, a, b):
        assert len(a) == len(b)
        shuffled_a = []
        shuffled_b = []
        permutation = np.random.permutation(len(a))
        for idx in permutation:
            shuffled_a.append(a[idx])
            shuffled_b.append(b[idx])
        return shuffled_a, shuffled_b

    def __iter__(self):
        return self

    def __next__(self):
        
        if (self.data_pointer >= len(self.inputs)):
            self.data_pointer = 0
            raise StopIteration
            
        end = min(len(self.inputs), self.data_pointer+self.batch_size)
        
        inputs = self.inputs[self.data_pointer:end]
        targets = self.targets[self.data_pointer:end]
        
        inp_batch, seqlen_batch, tar_batch = self.batchify(inputs, targets)
        
        self.data_pointer += self.batch_size
        
        return inp_batch, tar_batch, seqlen_batch
    

    def batchify(self, inputs, targets):
        
        assert len(inputs) == len(targets)

        N = len(inputs)
        seq_lengths = [len(x) for x in inputs]
        max_length = max(seq_lengths)

        sorted_inp, sorted_tar, sorted_seqlen = self.sort_data(inputs, targets, seq_lengths)

        padded_data = []
        for data, seqlen in zip(sorted_inp, sorted_seqlen):
            pad_len = max_length - seqlen

            if pad_len > 0:
                paddings = np.zeros((pad_len, self.d_embed))
                padded = np.concatenate((data, paddings), axis=0)
                padded_data.append(padded)
            else:
                padded_data.append(data)

        padded_data = np.array(padded_data)
        seq_lens = np.array(sorted_seqlen)
        targets = np.array(sorted_tar)

        assert padded_data.shape == (N, max_length, self.d_embed)
        assert len(seq_lens) == N
            
        return torch.tensor(padded_data, dtype=torch.float), torch.tensor(seq_lens, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def sort_data(self, inputs, targets, seq_lengths):
        sorted_data = sorted(zip(seq_lengths, range(len(inputs)), inputs, targets), reverse = True)

        i = [x for _,_,x,_ in sorted_data]
        t = [x for _,_,_,x in sorted_data]            
        seq_len = [x for x,_,_,_ in sorted_data]
        return i, t, seq_len


if __name__ == "__main__":
    # TEST
    d_embed = 1024
    D = [np.random.uniform(size=(random.randint(64,1000), d_embed)) for i in range(1000)]
    T = [random.randint(0,10) for i in range(1000)]

    dataloader = DataLoader(D, T, batch_size=64, d_embed=d_embed)
    for d in dataloader:

        print(d)