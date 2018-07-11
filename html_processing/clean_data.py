import pickle
import argparse

def char_counts(anime, c):
    return anime.title.count(c) + anime.synopsis.count(c)

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('vocab_path')
parser.add_argument('out_path')
args = parser.parse_args()

with open(args.data_path, 'rb') as f:
    data = pickle.load(f)

with open(args.vocab_path, 'rb') as f:
    vocab = pickle.load(f)

bad = []
for anime in data:
    vocab_chars = sum([char_counts(anime, c) for c in vocab])
    if vocab_chars != (len(anime.title) + len(anime.synopsis)):
        bad.append(anime)

data.difference_update(bad)

with open(args.out_path, 'wb') as f:
    pickle.dump(data, f)

print('cleaned {}. {} items removed.'.format(args.data_path, len(bad)))

