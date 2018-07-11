#!/bin/bash

python train_on_char_data.py ../data/Action_data/Action.dataset ../data/Action_data/Action.vocab \
    --epochs 50 \
    --batch_size 32 \
    --lstm_n_layers 2 \
    --lstm_n_hidden 300 \
    --lstm_dropout .55 \
    --log_freq 1 \
    --lr_decay_type exponential \
    --lr_start 1e-3 \
    --test_pct 0.2 \
    --val_pct 0.2 \
    --model_name even_less_boring

