#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=8

DATA_PATH=/workspace/Megatron-LM/my-gpt2_text_document
#CHECKPOINT_PATH=<Specify path>


python pretrain_gpt.py \
       --num-layers 48 \
       --hidden-size 1600 \
       --num-attention-heads 25 \
       --micro-batch-size 4 \
       --global-batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100 \
       --lr-decay-iters 320 \
       --data-path $DATA_PATH \
       --vocab-file /workspace/Megatron-LM/examples/gpt2-vocab.json \
       --merge-file /workspace/Megatron-LM/examples/gpt2-merges.txt \
       --data-impl mmap \
       --split 800,100,100 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 100 \
       --eval-interval 10 \
       --eval-iters 10 \
       --fp16