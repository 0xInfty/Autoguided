device=0
refdir="04_Tiny_LR/Ref/00/network-snapshot-0005159-0.100.pkl"
# refdir="04_Tiny_LR/Ref/00/network-snapshot-0001279-0.100.pkl"
# refdir="04_Tiny_LR/Ref/00/network-snapshot-0000359-0.100.pkl"
guidanceshort=1.7; guidancelong=2.2


##################################################################
###### JUST THE MODELS AFTER TRAINING FOR 48 HS ##################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --no-class-metrics

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=22000 --max-epoch=22000 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=21500 --max-epoch=21500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=20500 --max-epoch=20500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics

##################################################################
###### ALL ON SAME EPOCH #########################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200 --no-class-metrics

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --min-epoch=18500 --max-epoch=18500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics

##################################################################
###### ALL ON SAME NIMG ##########################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=4000 --max-epoch=4000 --save-nimg=200 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200 --no-class-metrics

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=4000 --max-epoch=4000 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidanceshort --emas=0.05 --no-class-metrics

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --min-epoch=4000 --max-epoch=4000 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --min-epoch=3500 --max-epoch=3500 --save-nimg=200  --guide-path=$refdir --guidance-weight=$guidancelong --emas=0.1 --no-class-metrics


##################################################################
###### ALL VALIDATION CURVES #####################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=2000 --max-epoch=5000 --period=1000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=6000 --max-epoch=21500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=21500 --max-epoch=21500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2000 --max-epoch=5000 --period=1000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=6000 --max-epoch=21500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=21500 --max-epoch=21500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2000 --max-epoch=5000 --period=1000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=6000 --max-epoch=21500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=21500 --max-epoch=21500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --max-epoch=1000 --period=250
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=2000 --max-epoch=5000 --period=1000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=6000 --max-epoch=21500 --period=5000
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=21500 --max-epoch=21500

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

##################################################################
###### MORE DETAIL IN ACCURACY CURVES ############################
##################################################################

# 2500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=18500 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=20500 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=22000 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=21500 --period=5000 --shift

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=18500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=20500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=22000 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=21500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=18500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=20500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=22000 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=2500 --max-epoch=21500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05

# 1225
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=18500 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=20500 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=22000 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=21500 --period=5000 --shift

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=18500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=20500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=22000 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=21500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=18500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=20500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=22000 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=1225 --max-epoch=21500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05

# 3775
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=18500 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=20500 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=22000 --period=5000 --shift
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=21500 --period=5000 --shift

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=18500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=20500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=22000 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=21500 --period=5000 --shift --guidance-weight=$guidancelong --emas=0.1

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=18500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Early_AJEST/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=20500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=22000 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-fd-metrics --min-epoch=3775 --max-epoch=21500 --period=5000 --shift --guidance-weight=$guidanceshort --emas=0.05


##################################################################
###### MISSING CURVES ############################################
##################################################################

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --max-epoch=1000 --period=250 --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --min-epoch=2000 --max-epoch=21500 --period=5000 --emas=0.05
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --min-epoch=18500 --max-epoch=18500 --no-fd-metrics

# FIX FID/FD CURVES ##############################################
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-class-metrics --min-epoch=3500 --max-epoch=3500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-class-metrics --min-epoch=22000 --max-epoch=22000

# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/AJEST/00 --save-nimg=200 --no-class-metrics --min-epoch=18500 --max-epoch=18500
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-class-metrics --min-epoch=18500 --max-epoch=18500 --emas=0.1

# FIX, BUT LONG
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=04_Tiny_LR/Baseline/04 --save-nimg=200 --no-class-metrics --min-epoch=11500 --max-epoch=22000 --emas=0.1

# RUN IN THE OTHER PC
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-class-metrics --min-epoch=21500 --max-epoch=21500 --emas=0.1

# RUN IN THE OTHER PC, BUT LONG
# CUDA_VISIBLE_DEVICES=$device python Images/get_validation_metrics.py validation --models-dir=06_CorrectEMA/Random/00 --save-nimg=200 --no-class-metrics --min-epoch=6000 --max-epoch=21500 --emas=0.05