ref="/00_PreTrained"; subdir="/New"; series="EarlyACIDNonInverted"; device=1

# Test a single given model
python ToyExample/toy_example.py test --acid --seed 7 --net-path "ToyExample"$subdir"/"$series"/iter4096learner.pkl" --ema-path "ToyExample"$subdir"/"$series"/iter4096.pkl" --guide-path "ToyExample"$ref"/iter0512.pkl" --logging "ToyExample"$subdir"/log_"$series".txt" --device $device