ref="/00_PreTrained"; subdir="/99_ACIDParams"; series="EarlyACIDNonInverted"; device=2

# Change n, compare to the default value n = 16 
python ToyExample/toy_example.py train --n 4 --outdir "ToyExample"$subdir"/"$series"_N_4_F_0.80_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_4_F_0.80_L_1.txt" --device $device
python ToyExample/toy_example.py train --n 8 --outdir "ToyExample"$subdir"/"$series"_N_8_F_0.80_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_8_F_0.80_L_1.txt" --device $device
python ToyExample/toy_example.py train --n 32 --outdir "ToyExample"$subdir"/"$series"_N_32_F_0.80_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_32_F_0.80_L_1.txt" --device $device

# Change filter_ratio, compare to the default value filter_ratio = 0.8
python ToyExample/toy_example.py train --filt 0.65 --outdir "ToyExample"$subdir"/"$series"_N_16_F_0.65_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_16_F_0.65_L_1.txt" --device $device
python ToyExample/toy_example.py train --filt 0.50 --outdir "ToyExample"$subdir"/"$series"_N_16_F_0.50_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_16_F_0.50_L_1.txt" --device $device
python ToyExample/toy_example.py train --filt 0.35 --outdir "ToyExample"$subdir"/"$series"_N_16_F_0.35_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_16_F_0.35_L_1.txt" --device $device
python ToyExample/toy_example.py train --filt 0.20 --outdir "ToyExample"$subdir"/"$series"_N_16_F_0.20_L_1" --early --acid --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --seed 0 --verbose --logging "ToyExample"$subdir"/log_"$series"_N_16_F_0.20_L_1.txt" --device $device
