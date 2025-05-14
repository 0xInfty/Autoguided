import os
import torch

import pyvdirs.dirs as dirs
import sys
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ToyExample"))

from ToyExample.toy_example import do_test
import pyvtools.text as vtext

# %%

test_batch_size = 2**14
series = ["18_Statistics", "19_ACIDParams", "21_Repetitions", "23_NormalizedLogits"]

get_path = lambda series : os.path.join(dirs.MODELS_HOME, "ToyExample", series)

# %%

for s in series:

    series_path = get_path(s)
    contents = os.listdir(series_path)
    folders = [c for c in contents if os.path.isdir(os.path.join(series_path, c))]
    folders = vtext.filter_by_string_must(folders, ["Failed", "Old"], must=False)
    log_files = ["log_"+f+".txt" for f in folders]
    assert all([os.path.isfile(os.path.join(series_path, f)) for f in log_files]), "Some logs have not been found"

    for folder, log_file in zip(folders, log_files):

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
        else:
            guide_filepath = None

        # %%

        do_test(net_filepath, ema_path=EMA_filepath, guide_path=guide_filepath, acid=acid, 
                classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
                n_samples=test_batch_size, batch_size=test_batch_size, test_outer=True,
                guidance_weight=3,
                seed=0, generator=None,
                device=torch.device('cuda'),
                log_filename=log_filepath)