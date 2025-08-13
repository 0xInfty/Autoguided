device=0
refdir="04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl"
guidanceshort=1.7; guidancelong=2.2

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/AJEST/02 --save-nimg=200
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Random/00 --save-nimg=200
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/AJEST/02 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/AJEST/02 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidaguidancelongnceshort --emas=0.1

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --seeds=0-199 --save-nimg=200 --max-epoch=500 --no-wandb --emas=0.05 --guide-path=$refdir --guidance-weight=1.7
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --seeds=0-199 --save-nimg=200 --max-epoch=500 --no-wandb --emas=0.1 --guide-path=$refdir --guidance-weight=2.2
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --seeds=0-199 --save-nimg=200 --max-epoch=500 --no-wandb