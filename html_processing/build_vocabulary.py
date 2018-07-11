import argparse
import pickle

from vocabulary import Vocabulary

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('out_path')
    parser.add_argument('--size', type=int, default=80)
    parser.add_argument('--use_all', action='store_true')
    args = parser.parse_args()

    if args.use_all:
        args.size = None

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    vocab = Vocabulary(data, args.size)

    with open(args.out_path, 'wb') as f:
        pickle.dump(vocab.mapping, f)

    print('dumped vocab to ' + str(args.out_path) + '.')

