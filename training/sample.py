import argparse
import pickle
import random
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as S
from networks import CharPredictor, one_hot_converter


def char_to_batch_tensor(char, embedding):
    return embedding(torch.LongTensor([[char]]))


def next_char_probs(net, char, embedding, temperature):
    out = net(char_to_batch_tensor(char, embedding), init_hiddens=False)
    flat = temperature * out[0, -1]
    return F.softmax(flat, dim=0)


def sample_char_from_network(net, char, embedding, temperature, sample_complete, sample_max):
    forbidden = embedding.weight.data.size(0) - 1
    next_probs = next_char_probs(net, char, embedding, temperature)
    if sample_max:
        return next_probs.argmax()
    sampler = S.WeightedRandomSampler(next_probs, num_samples=1)
    samples = sampler.__iter__()
    for sample in samples:
        if sample.item() != forbidden or sample_complete:
            return sample
    return torch.LongTensor([random.randint(0, forbidden - 1)])


def sample_from_network(net, first_char, n_samples, embedding, temperature, sample_complete, sample_max):
    chars = [first_char]
    if not sample_complete:
        for _ in range(n_samples):
            chars.append(
                    sample_char_from_network(
                        net, 
                        chars[-1], 
                        embedding,
                        temperature,
                        sample_complete,
                        sample_max,
                    )
            )
    else:
        end_char = embedding.weight.data.size(0) - 1
        while chars[-1] != end_char:
            chars.append(
                    sample_char_from_network(
                        net, 
                        chars[-1], 
                        embedding,
                        temperature,
                        sample_complete,
                        sample_max,
                    )
            )

    return chars


def to_text(sample, backward_vocab):
    return ''.join([backward_vocab[char.item()] for char in sample])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('vocab_path')
    parser.add_argument('--primer_text')
    parser.add_argument('--sample_length', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--sample_complete', action='store_true')
    parser.add_argument('--sample_max', action='store_true')

    args = parser.parse_args()
    with open(args.checkpoint_path, 'rb') as ckpt_file:
        checkpoint = pickle.load(ckpt_file)

    with open(args.vocab_path, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
        vocab_backward = {vocab[k]: k for k in vocab}
        make_one_hot = one_hot_converter(vocab)

    if args.primer_text is None:
        args.primer_text = vocab_backward[random.randint(0, len(vocab) - 1)]
    primer = torch.LongTensor([[vocab[char] for char in args.primer_text]])
    net = CharPredictor.from_checkpoint(checkpoint)
    output = net(make_one_hot(primer)) # prime hidden layers
    probs = F.softmax(output[0, -1], dim=0)
    sampler = S.WeightedRandomSampler(probs, num_samples=1)
    first_char = next(sampler.__iter__())
    sample = sample_from_network(net, first_char, args.sample_length, make_one_hot, args.temperature, args.sample_complete, args.sample_max)
    if args.sample_complete:
        sample = sample[:-1]
    print(args.primer_text + to_text(sample, vocab_backward))

