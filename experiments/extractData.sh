#!/bin/bash
if [ -z "$1" ]; then
	echo usage: $0 logdir
	exit
fi
if [ -n "$2" ]; then
	target=$2
else
	target=data
fi

echo logdir $1
python serialize_tensorboard.py --logdir "$1" --target $target --overwrite