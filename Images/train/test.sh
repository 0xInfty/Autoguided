preset="edm2-cifar10-xxs"; dataset="cifar10"
subdir="01_CIFAR10/test"; ref="network-snapshot-0025165-0.050.pkl"
batchsize=512
devices=1,2; ndevices=2

# With no data selection
# CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize

# With data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref