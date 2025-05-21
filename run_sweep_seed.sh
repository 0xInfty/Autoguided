ref="/00_PreTrained"; subdir="/24_Statistics"; device=0

#######################################################

# No ACID, for comparison
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_000.txt" --device $device
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_073.txt" --device $device
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_172.txt" --device $device
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_231.txt" --device $device
# python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --no-acid --outdir "ToyExample"$subdir"/NoACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_NoACID_357.txt" --device $device

#######################################################

# ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --acid --outdir "ToyExample"$subdir"/ACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_ACID_357.txt" --device $device

# Non-Inverted ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --acid --outdir "ToyExample"$subdir"/ACIDNonInverted_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_ACIDNonInverted_357.txt" --device $device

# Early ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --acid --outdir "ToyExample"$subdir"/EarlyACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --acid --outdir "ToyExample"$subdir"/EarlyACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --acid --outdir "ToyExample"$subdir"/EarlyACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --acid --outdir "ToyExample"$subdir"/EarlyACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --acid --outdir "ToyExample"$subdir"/EarlyACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACID_357.txt" --device $device

# Early Non-Inverted ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --acid --outdir "ToyExample"$subdir"/EarlyACIDNonInverted_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyACIDNonInverted_357.txt" --device $device

# Late ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --acid --outdir "ToyExample"$subdir"/LateACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --acid --outdir "ToyExample"$subdir"/LateACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --acid --outdir "ToyExample"$subdir"/LateACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --acid --outdir "ToyExample"$subdir"/LateACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --acid --outdir "ToyExample"$subdir"/LateACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACID_357.txt" --device $device

# Late Non-Inverted ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --acid --outdir "ToyExample"$subdir"/LateACIDNonInverted_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_LateACIDNonInverted_357.txt" --device $device

#######################################################

# Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/InterpolACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACID_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/InterpolACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACID_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/InterpolACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACID_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/InterpolACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACID_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --invert --interpol --acid --outdir "ToyExample"$subdir"/InterpolACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACID_357.txt" --device $device

# Non-Inverted Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/InterpolACIDNonInverted_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACIDNonInverted_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/InterpolACIDNonInverted_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACIDNonInverted_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/InterpolACIDNonInverted_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACIDNonInverted_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/InterpolACIDNonInverted_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACIDNonInverted_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --interpol --acid --outdir "ToyExample"$subdir"/InterpolACIDNonInverted_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_InterpolACIDNonInverted_357.txt" --device $device

# Early Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACID_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACID_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACID_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACID_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --invert --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACID_357.txt" --device $device

# Early Non-Inverted Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACIDNonInverted_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACIDNonInverted_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACIDNonInverted_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACIDNonInverted_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACIDNonInverted_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACIDNonInverted_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACIDNonInverted_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACIDNonInverted_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --early --interpol --acid --outdir "ToyExample"$subdir"/EarlyInterpolACIDNonInverted_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_EarlyInterpolACIDNonInverted_357.txt" --device $device

# Late Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACID_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACID_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACID_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACID_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACID_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACID_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACID_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACID_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --invert --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACID_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACID_357.txt" --device $device

# Late Non-Inverted Interpol ACID --> From the beginning, for as long as it runs
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACIDNonInverted_000" --seed 0 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACIDNonInverted_000.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACIDNonInverted_073" --seed 73 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACIDNonInverted_073.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACIDNonInverted_172" --seed 172 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACIDNonInverted_172.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACIDNonInverted_231" --seed 231 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACIDNonInverted_231.txt" --device $device
python ToyExample/toy_example.py train --guidance --guide-path "ToyExample"$ref"/Ref/iter0512.pkl" --late --interpol --acid --outdir "ToyExample"$subdir"/LateInterpolACIDNonInverted_357" --seed 357 --val --test --verbose --logging "ToyExample"$subdir"/log_LateInterpolACIDNonInverted_357.txt" --device $device
