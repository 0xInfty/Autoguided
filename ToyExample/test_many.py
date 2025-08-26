import os
main_dir = os.path.dirname(os.path.dirname(os.getcwd()))
os.chdir(main_dir)

import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ToyExample"))
from socket import gethostname

import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from ToyExample.toy_example import do_test
import pyvtools.text as vtext
import ours.utils as utils

# series = ["18_Statistics", "19_ACIDParams", "21_Repetitions", "23_NormalizedLogits"]
# series = ["25_ACIDParams_00"]
# series = ["23_NormalizedLogits"]
# series = ["28_Test_Size"]
series = ["29_Statistics"]

# results_filename = "TestResults_25_ACIDParams_00.json"
# results_filename = "TestResults_23.json"
# results_filename = "TestResults_28.json"
results_filename = "TestResults_29.json"

def test_many(series, results_filename="TestResults.json", test_batch_size=10*2**14, test_n_samples=10*2**14, test_seed=7):

    get_path = lambda series : os.path.join(dirs.MODELS_HOME, "ToyExample", series)

    host_id = gethostname()
    other_hosts = vtext.filter_by_string_must(list(dirs.check_directories_file().keys()), [host_id,"else"], must=False)

    results_filepath = os.path.join(dirs.RESULTS_HOME, "ToyExample", results_filename)

    series_folders = {}
    for s in series:
        series_path = get_path(s)
        contents = os.listdir(series_path)
        folders = [c for c in contents if os.path.isdir(os.path.join(series_path, c))]
        folders = vtext.filter_by_string_must(folders, ["Failed", "Old", "Others"], must=False)
        series_folders[s] = folders

    test_results = {}
    for s in series:

        series_path = get_path(s)
        log_files = ["log_"+f+".txt" for f in series_folders[s]]
        assert all([os.path.isfile(os.path.join(series_path, f)) for f in log_files]), "Some logs have not been found"

        test_results[s] = {}
        for folder, log_file in zip(series_folders[s], log_files):

            log_filepath = os.path.join(series_path, log_file)

            files = os.listdir(os.path.join(series_path, folder))
            net_file = vtext.filter_by_string_must(files, "learner")[0]
            EMA_file = "".join(net_file.split("learner"))

            net_filepath = os.path.join(series_path, folder, net_file)
            EMA_filepath = os.path.join(series_path, folder, EMA_file)

            with open(log_filepath, "r") as f:
                acid = False
                for i, line in enumerate(f):
                    if "ACID = True" in line:
                        acid = True
                    if "Guide model loaded from" in line or i>70: 
                        break
            if "Guide model loaded from" in line:
                guide_line = line
                guide_filepath = guide_line.split("Guide model loaded from ")[-1].split("\n")[0]
                for h in other_hosts:
                    guide_filepath = guide_filepath.replace(dirs.check_directories_file()[h]["models_home"], dirs.MODELS_HOME)
            else:
                guide_filepath = None

            folder_results = do_test(
                net_filepath, ema_path=EMA_filepath, guide_path=guide_filepath, acid=acid, 
                classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
                n_samples=test_n_samples, batch_size=test_batch_size, 
                test_outer=True, test_mandala=True,
                guidance_weight=3,
                seed=test_seed, generator=None,
                log_filename=log_filepath,
                device=torch.device('cuda'))
            
            test_results[s][folder] = folder_results

            with open(results_filepath, "w") as file:
                json.dump({"test_n_samples":test_n_samples,
                        "test_batch_size":test_batch_size,
                        "test_seed":test_seed,
                        **test_results}, 
                        file)