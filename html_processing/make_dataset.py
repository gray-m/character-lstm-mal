import math
import argparse
import pickle
import sys

# would be better to use one or the other but i don't care
# this is a shitpost
import torch
import torch.utils.data
import numpy as np


def split_string(string, length):
    n_subs = int(math.ceil(len(string)/length))
    subs = []
    start = 0
    end = min(length, len(string))
    for _ in range(n_subs):
        subs.append(string[start:end])
        start += length
        end = min(start + length, len(string))
    return subs


def string_to_category_list(string, vocab_map, seq_len):
    filler = len(vocab_map)
    suffix = [filler]*(seq_len - len(string)) # pad the end 
    return [vocab_map[c] for c in string] + suffix 

# will return a (seq_length, vocab_length vector)
# REMEMBER to use batch_first=True
def make_one_hot(string, seq_length, vocab_map):
    oh_string = np.zeros(shape=(seq_length, len(vocab_map)), dtype=np.float32)
    for i, char in enumerate(string):
        oh_string[i, vocab_map[char]] = 1.
    return oh_string


# oh_data will be (n_strings, string_length, vocab_length)
def make_datasets(oh_data, val_proportion):
    n_val = int(oh_data.shape[0]*val_proportion)
    n_train = oh_data.shape[0] - n_val
    examples = oh_data[..., :-1]
    labels = oh_data[..., 1:]
    train_examples = torch.FloatTensor(examples[:n_train])
    train_labels = torch.FloatTensor(labels[:n_train])
    val_examples = torch.FloatTensor(examples[n_train:])
    val_labels = torch.FloatTensor(labels[n_train:])

    train_dataset = torch.utils.data.TensorDataset(train_examples, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_examples, val_labels)

    return train_dataset, val_dataset


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('vocab_path')
    parser.add_argument('out_path')
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--val_proportion', type=float, default=1/5)
    parser.add_argument('--print_sizes', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        print('reading data from', args.data_path)
        data = pickle.load(f)
        if args.print_sizes:
            print('data size:', sum([sys.getsizeof(a.synopsis) for a in data]))

    with open(args.vocab_path, 'rb') as f:
        print('reading vocab from', args.vocab_path)
        vocab_map = pickle.load(f)

    print('splitting strings into substrings of length', args.seq_len)
    strings = []
    for anime in data:
        strings.extend(split_string(anime.synopsis, args.seq_len))

    encoded = [string_to_category_list(string, vocab_map, args.seq_len) for string in strings]
    assert all([len(enc) == args.seq_len for enc in encoded])
    inputs = torch.LongTensor([enc[:-1] for enc in encoded])
    labels = torch.LongTensor([enc[1:] for enc in encoded])
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    if args.print_sizes:
        print('size after splitting:', sum([sys.getsizeof(s) for s in strings]))

    if not args.dry_run:
        with open(args.out_path, 'wb') as f:
            pickle.dump(dataset, f)

    print('dumped data to {}.'.format(args.out_path))

