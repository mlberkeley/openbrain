#!/bin/bash
if [ -z "$1" ]; then
	echo usage: $0 logdir
	exit
fi
if [ ! -f serialize_tensorboard.py ]; then
	wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/tensorboard/scripts/serialize_tensorboard.py
fi

echo logdir $1
python serialize_tensorboard.py --logdir "$1" --target data --overwrite