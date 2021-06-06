#!/bin/sh

python go_transformer.py --batch 128 --num-enc 3 --num-dec 3 --emb-dim 256 --ffn-dim 512 --head 8 --dropout 0.3 --epoch 40 --lr 0.0001

