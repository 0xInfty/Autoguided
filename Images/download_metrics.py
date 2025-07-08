import pyvdirs.dirs as dirs
import sys
import os
sys.path.insert(0, dirs.SYSTEM_HOME)
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "karras"))
sys.path.insert(0, os.path.join(dirs.SYSTEM_HOME, "ours"))

import csv
import wandb
import tqdm

api = wandb.Api()

def download_metrics(run_ids, output_filepath):

    history = []
    for run_id in run_ids:
        run = api.run(f"ajest/Images/{run_id}")
        hist = run.scan_history(keys=["Epoch", "Indices", "Selected indices"], page_size=100)
        history.append(hist)

    # Open the CSV file in write mode
    with open(output_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Rank", "Epoch", "Round", "Image ID", "Selected"])
        
        for epoch_i, rank_rows in enumerate(tqdm.tqdm(zip(*history))):
            for round_i in range(len(rank_rows[0]["Indices"])):
                for image_i in range(len(rank_rows[0]["Indices"][round_i])):
                    for rank_i in range(2):
                        img_id = rank_rows[rank_i]["Indices"][round_i][image_i]
                        is_img_id_selected = image_i in rank_rows[rank_i]["Selected indices"][round_i]
                        writer.writerow([rank_i, rank_rows[rank_i]["Epoch"], round_i, img_id, int(is_img_id_selected)])

if __name__=="__main__":

    run_ids = ["4giyu6ty", "nue64z6i"] # Each group has two runs, one per GPU
    filepath = os.path.join(dirs.MODELS_HOME, "Images/03_TinyImageNet/AJEST/00", "indices.csv")
    download_metrics(run_ids, filepath)