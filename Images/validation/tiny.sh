device=0

CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py --models-dir=04_Tiny_LR/Baseline/00 --save-nimg=200
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py --models-dir=04_Tiny_LR/AJEST/02 --save-nimg=200
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py --models-dir=04_Tiny_LR/Random/00 --save-nimg=200
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py --models-dir=04_Tiny_LR/Early_AJEST/05 --save-nimg=200