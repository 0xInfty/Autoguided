ref="/00_PreTrained"; subdir="/25_Repetitions"

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_00" --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_00.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_01" --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_01.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_02" --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_02.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_03" --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_03.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_04" --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_04.txt"

python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_00" --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_00.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_01" --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_01.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_02" --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_02.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_03" --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_03.txt"
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_04" --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_04.txt"