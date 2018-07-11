import argparse
import pickle

import os
# sketchy importing so pickle is happy
import sys
sys.path.append('../html_processing')

from tqdm import tqdm

import torch.nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from networks import CharPredictor, one_hot_converter, save_model


def training_progress(gen):
    return tqdm(gen, desc='epoch', position=0)


def epoch_progress(gen):
    return tqdm(gen, desc='train batch', position=1)


def validation_progress(gen):
    return tqdm(gen, desc='val batch', position=2)


def test_progress(gen):
    return tqdm(gen, desc='test batch', position=3)


def linear_decayed_learning_rate(start_lr, end_lr, steps, delay):
    def _lr(t):
        if t < delay:
            return start_lr
        t = t - delay
        return start_lr + (end_lr - start_lr) * (t/steps)
    return _lr


def exponential_decayed_learning_rate(start_lr, decay_rate, delay):
    def _lr(t):
        if t < delay:
            return start_lr
        t = t - delay
        return start_lr * (decay_rate**t)
    return _lr


def pass_through_dataset(net, dataset, objective, optimizer, embedding, max_grad_norm, train=True, test=False):
    total_loss = 0.
    if train and test:
        raise ValueError('you can\'t train and test at the same time, ya doofus')

    wrapper = epoch_progress
    if test:
        wrapper = test_progress
    elif not train:
        wrapper = validation_progress
    
    for batch in wrapper(dataset):
        inpt, expected = batch
        output = net(embedding(inpt))
        emb_expected = embedding(expected)
        loss = 0.
        for i in range(len(inpt[0])):
            loss = loss + objective(output[:, i], expected[:, i])
            total_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
    return total_loss/len(dataset)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('vocab_path')
    parser.add_argument('--model_init_path', help='path to checkpoint to init model from')
    parser.add_argument('--lstm_n_layers', type=int, default=2)
    parser.add_argument('--lstm_n_hidden', type=int, default=300)
    parser.add_argument('--lstm_dropout', type=float, default=0.3)
    parser.add_argument('--model_name', help='prefix to checkpoints in the checkpoints directory', default='boring')
    parser.add_argument('--grad_clip_value', type=float, default=5.)
    parser.add_argument('--lr_decay_type', choices=['linear', 'exponential'], default='linear')
    parser.add_argument('--lr_start', type=float, default=1e-3, help='adam optimizer learning rate before decay')
    parser.add_argument('--lr_end', type=float, default=5e-5, help='adam optimizer learning rate after linear decay')
    parser.add_argument('--lr_decay_delay', type=int, default=10, help='number of epochs to train for before decaying the learning rate')
    parser.add_argument('--lr_decay_epochs', type=int, default=25, help='number of epochs to decay learning rate over')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='decay rate used when exponential learning rate decay is enabled')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train network for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--val_pct', type=float, default=1./10, help='percent of training set to be used for validation')
    parser.add_argument('--test_pct', type=float, default=1./5, help='percent of total data set to be used as the test set')
    parser.add_argument('--log_freq', type=int, default=5, help='number of epochs between logs')
    
    args = parser.parse_args()

    with open(args.data_path, 'rb') as data_f:
        dataset = pickle.load(data_f)

    n_test = int(len(dataset)*args.test_pct)
    n_val = int((len(dataset) - n_test)*args.val_pct)

    # jank
    test_set = Subset(dataset, list(range(n_test)))
    val_set = Subset(dataset, [n_test + i for i in range(n_val)])
    train_set = Subset(dataset, [n_test + n_val + i for i in range(len(dataset) - n_test - n_val)])

    with open(args.vocab_path, 'rb') as vocab_f:
        vocab = pickle.load(vocab_f)
        make_one_hot = one_hot_converter(vocab)

    if args.lr_decay_type == 'linear':
        decayed_lr = linear_decayed_learning_rate(args.lr_start, args.lr_end, args.lr_decay_epochs, args.lr_decay_delay)
    elif args.lr_decay_type == 'exponential':
        decayed_lr = exponential_decayed_learning_rate(args.lr_start, args.lr_decay_rate, args.lr_decay_delay)

    objective = torch.nn.NLLLoss()

    if args.model_init_path is not None:
        with open(args.model_init_path, 'rb') as ckpt_f:
            checkpoint = pickle.load(ckpt_f)
        
        if checkpoint['net_params']['vocab_size'] != len(vocab):
            raise ValueError('Network has incorrect vocab size.')

        net = CharPredictor.from_checkpoint(checkpoint)
        optimizer = Adam(net.parameters(), lr=decayed_lr(0))
        optimizer.load_state_dict(checkpoint['opt_state'])
    else:
        net = CharPredictor(vocab_size=len(vocab),
                            lstm_n_hidden=args.lstm_n_hidden,
                            lstm_n_layers=args.lstm_n_layers,
                            lstm_dropout=args.lstm_dropout)
        optimizer = Adam(net.parameters(), lr=decayed_lr(0))

    save_path = os.path.join('checkpoints', args.model_name + '.ckpt')

    if os.path.exists('train_log.txt'):
        os.remove('train_log.txt')
    
    min_val_loss = 1e9 # eh, big enough

    for epoch in training_progress(range(args.epochs)):
        train_loader = DataLoader(
                          train_set,
                          batch_size=args.batch_size,
                          shuffle=True,
                       )
        val_loader = DataLoader(
                        val_set,
                        batch_size=args.batch_size,
                        shuffle=True,
                     )

        train_loss = pass_through_dataset(
                        net,
                        train_loader,
                        objective,
                        optimizer,
                        make_one_hot,
                        args.grad_clip_value,
                    )
            
        val_loss = pass_through_dataset(
                      net,
                      val_loader,
                      objective,
                      optimizer,
                      make_one_hot,
                      args.grad_clip_value,
                      train=False,
                   )
        
        if val_loss < min_val_loss:
            with open('train_log.txt', 'a') as train_log:
                print('Validation loss decreased at epoch %d; saving model.\n'%(epoch),
                        file=train_log)

            save_model(net, optimizer, save_path)
            min_val_loss = val_loss

        if epoch%args.log_freq == 0:
            with open('train_log.txt', 'a') as train_log:
                print('Train loss at epoch %d: %.4f.'%(epoch, train_loss),
                        file=train_log)
                print('Validation loss at epoch %d: %.4f.\n'%(epoch, val_loss),
                        file=train_log)

        for param_group in optimizer.param_groups:
            param_group['lr'] = decayed_lr(epoch)


    test_loader = DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                 )
    test_loss = pass_through_dataset(
                   net,
                   test_loader,
                   objective,
                   optimizer,
                   make_one_hot,
                   args.grad_clip_value,
                   train=False,
                   test=True,
                )

    with open('train_log.txt', 'a') as train_log:
        print('Test loss after %d epochs: %.4f.\n'%(args.epochs, test_loss),
                file=train_log)

