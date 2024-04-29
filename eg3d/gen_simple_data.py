"""
This code is to sample some frames from a face dataset.
The face dataset should have structure below:
| - Root/
| - - frames/
| - - masks/
| - - cam_params.pt
| - - latent_z.pt
| - - latent_w.pt
Example:
python gen_simple_data.py --origin_data_root ~/data/capvideo/IMG_0766/ --output_path ~/data/capvideo_simple/IMG_0766_32f/ --num_frames 32
"""

import os
import glob
import argparse
import shutil
import random
import pdb

import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_data_root', type=str, default='./', help='path to video frames')
    parser.add_argument('--output_path', type=str, default='./', help='output path')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames for the output.')
    opts = parser.parse_args()

    frame_list = sorted(glob.glob(os.path.join(opts.origin_data_root, 'frames', '*png')) + glob.glob(os.path.join(opts.origin_data_root, 'frames', '*jpg')) + glob.glob(os.path.join(opts.origin_data_root, 'frames', '*jpeg')))
    mask_list = sorted(glob.glob(os.path.join(opts.origin_data_root, 'masks', '*png')) + glob.glob(os.path.join(opts.origin_data_root, 'masks', '*jpg')) + glob.glob(os.path.join(opts.origin_data_root, 'frames', '*jpeg')))

    idx = random.sample(range(len(frame_list)), k=opts.num_frames)
    idx = sorted(idx)
    frame_list = list(np.array(frame_list)[idx])  # to fix: kinda inefficient
    if len(mask_list) > 0:
        mask_list = list(np.array(mask_list)[idx]) 

    os.makedirs(os.path.join(opts.output_path, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(opts.output_path, 'masks'), exist_ok=True)
    
    # Camera parameters
    import json
    cam_path = os.path.join(opts.origin_data_root, 'dataset.json')
    with open(cam_path, 'r') as f:
        cam_params = json.load(f)
    f.close()
    cam_params = np.array(cam_params['labels'])
    cam_params_selected = cam_params[idx]
    cam_params_selected = [list(cam_param_selected) for cam_param_selected in cam_params_selected]
    cam_params_selected = {'labels': cam_params_selected}
    with open(os.path.join(opts.output_path, 'dataset.json'), 'w') as f:
        json.dump(cam_params_selected, f, indent=4)
    f.close()

    

    if len(mask_list) > 0:
        for frame_path, mask_path in zip(frame_list, mask_list):
            assert os.path.basename(frame_path) == os.path.basename(mask_path), "frame and mask do not match!"
            
            # copy frames and masks
            shutil.copyfile(frame_path, os.path.join(opts.output_path, 'frames', os.path.basename(frame_path)))
            shutil.copyfile(mask_path, os.path.join(opts.output_path, 'masks', os.path.basename(mask_path)))
    else:
        for frame_path in frame_list:
            # copy frames and masks
            shutil.copyfile(frame_path, os.path.join(opts.output_path, 'frames', os.path.basename(frame_path)))

