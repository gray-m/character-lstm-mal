#!/bin/bash

genre=$1
dirname=$genre
dirname+="_data"
data_dir=../data/$dirname
seq_len=$2

if [ -d $data_dir ]; then
    rm -rf $data_dir
fi

mkdir $data_dir 

python consolidate_directory.py ../html_raw $data_dir/$genre.out \
    --genre=$genre \
    --verbose

python build_vocabulary.py $data_dir/$genre.out $data_dir/$genre.vocab

python clean_data.py $data_dir/$genre.out $data_dir/$genre.vocab $data_dir/$genre.clean

python make_dataset.py $data_dir/$genre.clean $data_dir/$genre.vocab $data_dir/$genre.dataset \
    --seq_len=$seq_len

