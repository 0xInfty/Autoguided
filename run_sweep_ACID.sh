# Remember to include your logging folder in the logging filepath

# Change n, compare to the default value n = 16 
python ToyExample/toy_example.py train --n 4 --outdir "ToyExample/19_ACIDParams/ACID_N_4_F_0.80_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_4_F_0.80_L_1.txt"
python ToyExample/toy_example.py train --n 8 --outdir "ToyExample/19_ACIDParams/ACID_N_8_F_0.80_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_8_F_0.80_L_1.txt"
python ToyExample/toy_example.py train --n 32 --outdir "ToyExample/19_ACIDParams/ACID_N_32_F_0.80_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_32_F_0.80_L_1.txt"

# Change filter_ratio, compare to the default value filter_ratio = 0.8
python ToyExample/toy_example.py train --filt 0.65 --outdir "ToyExample/19_ACIDParams/ACID_N_16_F_0.65_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_16_F_0.65_L_1.txt"
python ToyExample/toy_example.py train --filt 0.50 --outdir "ToyExample/19_ACIDParams/ACID_N_16_F_0.50_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_16_F_0.50_L_1.txt"
python ToyExample/toy_example.py train --filt 0.35 --outdir "ToyExample/19_ACIDParams/ACID_N_16_F_0.35_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_16_F_0.35_L_1.txt"
python ToyExample/toy_example.py train --filt 0.20 --outdir "ToyExample/19_ACIDParams/ACID_N_16_F_0.20_L_1" --invert --acid --guidance --guide-path "ToyExample/00_PreTrained/Ref/iter0512.pkl" --seed 0 --verbose --logging "SCID/ToyExample/19_ACIDParams/log_ACID_N_16_F_0.20_L_1.txt"
