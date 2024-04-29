import json
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to cam_params.pt", required=True)
    a.add_argument("--pathOut", help="path to dataset.json", required=True)
    a.add_argument("--files_folder", help="filenames", required=True)
    args = a.parse_args()
    print(args)

    cam_params = torch.load(args.pathIn)
    mats = cam_params['proj_mat'].numpy()  # [N, 25]

    dataset = {'labels':[]}

    filenames = sorted(os.listdir(args.files_folder))
    for i, filename in enumerate(tqdm(filenames)):
        label = mats[i].reshape(-1).tolist()
        dataset["labels"].append([filename, label])
    
    with open(os.path.join(args.pathOut, 'dataset.json'), "w") as f:
        json.dump(dataset, f, indent=4)