preset="test-training"; dataset="cifar10"
subdir="99_Test/Random/00"; ref="01_CIFAR10/Ref/00/network-snapshot-0000039-0.100.pkl"
batchsize=512
devices=0,1; ndevices=2
device=0

# With no data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py  --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize

# With AJEST data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --acid --guide-path $ref

# With random data selection
CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$ndevices Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --selection
# CUDA_VISIBLE_DEVICES=$device python Images/train_edm2.py --outdir=$subdir --dataset $dataset --preset=$preset --batch-gpu=$batchsize --selection