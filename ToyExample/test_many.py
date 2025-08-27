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

from ToyExample.toy_example import do_test, extract_results_from_log
import pyvtools.text as vtext

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

def test_efficiency(series, results_filename="TestResults.json", test_batch_size=10*2**14, test_n_samples=10*2**14, test_seed=7):

    get_path = lambda series : os.path.join(dirs.MODELS_HOME, "ToyExample", series)

    host_id = gethostname()
    other_hosts = vtext.filter_by_string_must(list(dirs.check_directories_file().keys()), [host_id,"else"], must=False)

    results_filepath = os.path.join(dirs.RESULTS_HOME, "ToyExample", results_filename)
    results_filepath_time = "_time".join(os.path.splitext(results_filepath))
    results_filepath_examples = "_examples".join(os.path.splitext(results_filepath))
    mid_filepath_time = "_temp".join(os.path.splitext(results_filepath_examples))
    mid_filepath_examples = "_temp".join(os.path.splitext(results_filepath_time))

    series_folders = {}
    for s in series:
        series_path = get_path(s)
        contents = os.listdir(series_path)
        folders = [c for c in contents if os.path.isdir(os.path.join(series_path, c))]
        folders = vtext.filter_by_string_must(folders, ["Failed", "Old", "Others"], must=False)
        series_folders[s] = folders

    time = {}
    examples_seen = {}
    all_log_files = {}
    all_checkpoint_epochs = {}
    for s in series:

        series_path = get_path(s)
        log_files = ["log_"+f+".txt" for f in series_folders[s]]
        assert all([os.path.isfile(os.path.join(series_path, f)) for f in log_files]), "Some logs have not been found"

        time[s] = {}
        examples_seen[s] = {}
        all_log_files[s] = {}
        all_checkpoint_epochs[s] = {}
        for folder, log_file in zip(series_folders[s], log_files):
            all_log_files[s][folder] = log_file

            curves = extract_results_from_log(os.path.join(series_path, log_file))
            examples_seen[s][folder] = curves["examples_seen"]
            if len(curves["training_time"]) == len(curves["examples_seen"]):
                time[s][folder] = curves["training_time"]
            else:
                time[s][folder] = curves["training_time"][0::2] #TODO: Remove this patch once the code is fixed

            checkpoint_filenames = os.listdir(os.path.join(series_path, folder))
            checkpoint_filenames = vtext.filter_by_string_must(checkpoint_filenames, ".pkl")
            checkpoint_filenames = vtext.filter_by_string_must(checkpoint_filenames, "learner", must=False)
            checkpoint_filenames.sort()
            checkpoint_epochs = [vtext.find_numbers(fname)[0]-1 for fname in checkpoint_filenames]
            all_checkpoint_epochs[s][folder] = checkpoint_epochs
        
    max_common_time = np.min( [[max(time[s][folder]) for folder in series_folders[s]] for s in series] )
    max_common_examples = np.min( [[max(examples_seen[s][folder]) for folder in series_folders[s]] for s in series] )
    print(f"Max Common Time {max_common_time:.2f} min", )
    print("Max Common Number of Examples", max_common_examples)
    if not np.all( [[len(time[s][folder])==len(time[series[0]][series_folders[series[0]][0]]) for folder in series_folders[s]] for s in series] ):
        print("This could fail if the runs are trained for different total number of epochs")
    
    eval_epoch_time = {}
    eval_epoch_examples = {}
    for s in series:
        eval_epoch_time[s] = {}
        eval_epoch_examples[s] = {}
        for folder in series_folders[s]:
            # if "EarlyACID" in folder: print(time[s][folder])
            eval_epoch = np.argmin( np.abs( np.array(time[s][folder] - max_common_time) ) )  # Epoch on which this run reaches the chosen time
            # if "EarlyACID" in folder: print(all_checkpoint_epochs[s][folder])
            closest_epoch_index = np.argmin( np.abs( np.array(all_checkpoint_epochs[s][folder] - eval_epoch) ) ) # Closest existing epoch's indes
            eval_epoch_time[s][folder] = all_checkpoint_epochs[s][folder][ closest_epoch_index ] # Closest existing epoch
            # if "EarlyACID" in folder: print(closest_epoch_index, all_checkpoint_epochs[s][folder][ closest_epoch_index ])
            print(">", s, folder, ">>> Time >>>", eval_epoch_time[s][folder], ">>>", time[s][folder][ eval_epoch_time[s][folder] ])
            # if "EarlyACID" in folder: return
            
            eval_epoch = np.argmin( np.abs( np.array(examples_seen[s][folder] - max_common_examples) ) )  # Epoch on which this run reaches the chosen number of examples
            closest_epoch_index = np.argmin( np.abs( np.array(all_checkpoint_epochs[s][folder] - eval_epoch) ) ) # Closest existing epoch's indes
            eval_epoch_examples[s][folder] = all_checkpoint_epochs[s][folder][closest_epoch_index] # Closest existing epoch
            print(">", s, folder, ">>> Examples >>>", eval_epoch_examples[s][folder], ">>>", examples_seen[s][folder][ eval_epoch_examples[s][folder] ])
    
    test_results_time = {}
    test_results_examples = {}
    for s in series:
        series_path = get_path(s)

        print("> Starting with ", s)
        test_results_time[s] = {}
        test_results_examples[s] = {}
        for folder in series_folders[s]:

            log_filepath = os.path.join(series_path, all_log_files[s][folder])

            files = os.listdir(os.path.join(series_path, folder))
            files = vtext.filter_by_string_must(files, ".pkl")
            files = vtext.filter_by_string_must(files, "learner", must=False)
            checkpoint_filename_time = vtext.filter_by_string_must(files, str(eval_epoch_time[s][folder]+1))[0]
            checkpoint_filename_examples = vtext.filter_by_string_must(files, str(eval_epoch_examples[s][folder]+1))[0]

            checkpoint_filepath_time = os.path.join(series_path, folder, checkpoint_filename_time)
            checkpoint_filepath_examples = os.path.join(series_path, folder, checkpoint_filename_examples)

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

            print(">>> Working on time for", folder)
            results_time = do_test(
                checkpoint_filepath_time, guide_path=guide_filepath, acid=selection, 
                classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
                n_samples=test_n_samples, batch_size=test_batch_size, 
                test_outer=True, test_mandala=True,
                guidance_weight=3,
                seed=test_seed, generator=None,
                log_filename=log_filepath,
                device=torch.device('cuda'))
            test_results_time[s][folder] = results_time

            if s==series[-1] and folder==series_folders[s][-1]:
                save_filepath = results_filepath_time
            else:
                save_filepath = mid_filepath_time
            with open(save_filepath, "w") as file:
                json.dump({"test_n_samples":test_n_samples, "test_batch_size":test_batch_size, 
                           "test_seed":test_seed, **test_results_time}, file)
            
            print(">>> Working on examples for", folder)
            results_examples = do_test(
                checkpoint_filepath_examples, guide_path=guide_filepath, acid=selection, 
                classes='A', P_mean=-2.3, P_std=1.5, sigma_max=5, depth_sep=5,
                n_samples=test_n_samples, batch_size=test_batch_size, 
                test_outer=True, test_mandala=True,
                guidance_weight=3,
                seed=test_seed, generator=None,
                log_filename=log_filepath,
                device=torch.device('cuda'))
            test_results_examples[s][folder] = results_examples
            
            if s==series[-1] and folder==series_folders[s][-1]:
                save_filepath = results_filepath_examples
            else:
                save_filepath = mid_filepath_examples
            with open(save_filepath, "w") as file:
                json.dump({"test_n_samples":test_n_samples, "test_batch_size":test_batch_size, 
                           "test_seed":test_seed, **test_results_examples}, file)

#----------------------------------------------------------------------------
# Command line interface.

@click.group()
def cmdline():
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

@cmdline.command()
@click.option('--series',                   help='Series of folders to run for', metavar='STR',                     type=str, multiple=True)
@click.option('--output', 'filename',       help='Output json filename', metavar='PATH|URL',                        type=str, default=None)
def epoch(series, filename):
    test_many(series, results_filename=filename);

@cmdline.command()
@click.option('--series',                   help='Series of folders to run for', metavar='STR',                     type=str, multiple=True)
@click.option('--output', 'filename',       help='Output json filename', metavar='PATH|URL',                        type=str, default=None)
def efficiency(series, filename):
    test_efficiency(series, results_filename=filename);

if __name__ == "__main__":
    cmdline()