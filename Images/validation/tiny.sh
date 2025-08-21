device=0
refdir="04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl"
# refdir="04_Tiny_LR/Ref/00/network-snapshot-0001279-0.100.pkl"
# refdir="04_Tiny_LR/Ref/00/network-snapshot-0000359-0.100.pkl"
guidanceshort=1.7; guidancelong=2.2


##################################################################
###### JUST THE MODELS AFTER TRAINING FOR 48 HS ##################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --emas=0.1

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --emas=0.05


# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1


# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --emas=0.1

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --emas=0.05


##################################################################
###### ALL ON SAME EPOCH #########################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=18500 --max-epoch=18500 --save-nimg=200
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1

##################################################################
###### ALL ON SAME NIMG ##########################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=4000 --max-epoch=4000 --save-nimg=200
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200 --emas=0.1
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200 --emas=0.05

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=4000 --max-epoch=4000 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=4000 --max-epoch=4000 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1
CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1


##################################################################
###### ALL VALIDATION CURVES #####################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --emas=0.1 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=1000 --max-epoch=22000 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=22000 --max-epoch=22000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --emas=0.1 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1000 --max-epoch=18500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=1000 --max-epoch=21500 --period=5000 --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=21500 --max-epoch=21500 --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1000 --max-epoch=1000 --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --emas=0.1 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1000 --max-epoch=20500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=20500 --max-epoch=20500

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=1000 --max-epoch=22000 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=22000 --max-epoch=22000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=1000 --max-epoch=18500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=1000 --max-epoch=21500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=21500 --max-epoch=21500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/02 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --min-epoch=1000 --max-epoch=22000 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --min-epoch=22000 --max-epoch=22000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --min-epoch=1000 --max-epoch=18500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --min-epoch=1000 --max-epoch=21500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-fd-metrics --min-epoch=21500 --max-epoch=21500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=1000 --max-epoch=20500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-fd-metrics --min-epoch=20500 --max-epoch=20500