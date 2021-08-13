import torch
import cybertron.ct2.modeling as ct2_model
from typing import Tuple
from tqdm import tqdm
import time
​
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
#import torch.optim as optim
import torch.multiprocessing as mp
​
from torch.nn.parallel import DistributedDataParallel as DDP
​
# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.
​
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
​
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
​
def cleanup():
    dist.destroy_process_group()
​
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
​
    VOCAB_SIZE = 50257
    SENTENCE_SIZE = 1024
    EMBEDDING_DIM = 1600
    NUM_HEADS = 25
    NUM_LAYERS = 48
    DROPOUT_PROB = 0.1
​
    effective_bs = 32
    per_device_bs = effective_bs // 8
    BATCH_SIZE = 1
​
    # create model and move it to GPU with id rank
    model = ct2_model.CT2Model(vocab_size = VOCAB_SIZE, dropout_prob = DROPOUT_PROB, context_length = SENTENCE_SIZE, embedding_dim = EMBEDDING_DIM, num_heads = NUM_HEADS, num_layers= NUM_LAYERS)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
​
    input_ids = torch.LongTensor(BATCH_SIZE, SENTENCE_SIZE).random_(0, VOCAB_SIZE).cuda().to(rank)
    labels = torch.LongTensor(BATCH_SIZE, SENTENCE_SIZE).random_(0, VOCAB_SIZE).cuda().to(rank)
    inputs = (labels, labels)
    run(model=ddp_model, inputs=inputs, batch_size=BATCH_SIZE, inference=False, num_of_microbatches=per_device_bs)
​
    cleanup()
​
​
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
​
def run(model, inputs: Tuple[torch.Tensor, ...], batch_size: int, inference: bool, num_of_microbatches: int, n_iterations: int = 10) -> None:
    #num_of_microbatches = 4
    #model.to('cuda:0')
    optim = torch.optim.AdamW(model.parameters(),
                                lr=1e-05,
                                betas=(0.9, 0.997),
                                weight_decay=0.01,
                                eps=1e-8)
    
    #model = model.cuda()
    model.cuda().half()
​
    # if isinstance(model(*cuda_inputs), samba.SambaTensor):
    #     raise RuntimeError("SambaTensor is not allowed for GPU perf benchmarking")
​
    print("Measuring training performance.")
    model = model.train()
    optim.zero_grad()
    pbar = tqdm(range(n_iterations), total=n_iterations, dynamic_ncols=True)
    cuda_inputs = []
    for inp in inputs:
            if inp is not None:
                if inp.dtype == torch.float32:
                    cuda_inputs.append(inp.cuda().half())
                else:
                    cuda_inputs.append(inp.cuda())
    start = time.time()
    for _ in pbar:
        model.zero_grad()
        for _ in range(num_of_microbatches):
            loss = model(*cuda_inputs).loss
            loss.backward(torch.ones_like(loss))
        optim.step()
    torch.cuda.synchronize()
    end = time.time()
    duration = end - start
    print ("Duration: ", duration)
    #import pdb; pdb.set_trace()
    throughput = n_iterations * (batch_size*num_of_microbatches) / duration
    latency = duration / n_iterations
    print(f'throughput: {throughput} samples/s, measured over {n_iterations} iterations. Average latency: {latency} s.')
​
def main():
    run_demo(demo_basic, 8)
​
if __name__ == "__main__":
    main()