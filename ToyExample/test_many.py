import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ToyExample"))

from socket import gethostname
import click
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from ToyExample.toy_example import do_test
import pyvtools.text as vtext
import ours.utils as utils

# series = ["29_Statistics"]

# results_filename = "TestResults_29.json"

def test_many(series, results_filename="TestResults.json", test_batch_size=10*2**14, test_n_samples=10*2**14, test_seed=7):

    get_path = lambda series : os.path.join(dirs.MODELS_HOME, "ToyExample", series)

    host_id = gethostname()
    other_hosts = vtext.filter_by_string_must(list(dirs.check_directories_file().keys()), [host_id,"else"], must=False)

    results_filepath = os.path.join(dirs.RESULTS_HOME, "ToyExample", results_filename)
    mid_filepath = "_temp".join(os.path.splitext(results_filepath))

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

        print("> Starting with ", s)
        test_results[s] = {}
        for folder, log_file in zip(series_folders[s], log_files):

            log_filepath = os.path.join(series_path, log_file)

            files = os.listdir(os.path.join(series_path, folder))
            net_file = vtext.filter_by_string_must(files, "learner")[0]
            EMA_file = "".join(net_file.split("learner"))

            net_filepath = os.path.join(series_path, folder, net_file)
            EMA_filepath = os.path.join(series_path, folder, EMA_file)

            with open(log_filepath, "r") as f:
                selection = False
                acid = False
                for i, line in enumerate(f):
                    if "Selection = True" in line:
                        selection = True
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

            print(">>> Working on ", folder)
            folder_results = do_test(
                net_filepath, ema_path=EMA_filepath, guide_path=guide_filepath, acid=selection, 
                classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
                n_samples=test_n_samples, batch_size=test_batch_size, 
                test_outer=True, test_mandala=True,
                guidance_weight=3,
                seed=test_seed, generator=None,
                log_filename=log_filepath,
                device=torch.device('cuda'))
            
            test_results[s][folder] = folder_results

            if s==series[-1] and folder==series_folders[s][-1]:
                save_filepath = results_filepath
            else:
                save_filepath = mid_filepath
            with open(save_filepath, "w") as file:
                json.dump({"test_n_samples":test_n_samples, "test_batch_size":test_batch_size, 
                           "test_seed":test_seed, **test_results}, file)


#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--series',                   help='Series of folders to run for', metavar='STR',                     type=str, multiple=True)
@click.option('--output', 'filename',       help='Output json filename', metavar='PATH|URL',                        type=str, default=None)

def cmdline(series, filename):
    test_many(series, results_filename=filename);

if __name__ == "__main__":
    cmdline()