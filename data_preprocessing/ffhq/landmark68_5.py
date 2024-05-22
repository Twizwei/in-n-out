import numpy as np
import os
import glob
import argparse
from tqdm import tqdm

def writeLandmarks(txt_path, landmarks, fmt='%1.1f'):
    return np.savetxt(txt_path, landmarks, fmt)

def landmark68_to_5(lm68_path):
    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    lm68s = np.load(lm68_path)
    lm5s = np.stack([lm68s[:,lm_idx[0],:],np.mean(lm68s[:,lm_idx[[1,2]],:],1),np.mean(lm68s[:,lm_idx[[3,4]],:],1),lm68s[:,lm_idx[5],:],lm68s[:,lm_idx[6],:]], axis = 1)
    lm5s = lm5s[:, [1,2,0,3,4], :]

    return lm5s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmark68_path', type=str, required=True, help='68 landmark path')
    parser.add_argument('--frame_root', type=str, default="", help='process folder')
    args = parser.parse_args()

    ldm5 = landmark68_to_5(args.landmark68_path)
    framename_list = sorted(glob.glob(os.path.join(args.frame_root, '*.png')) + glob.glob(os.path.join(args.frame_root, '*.jpeg')))
    output_root = os.path.join(args.frame_root, 'detections')
    os.makedirs(output_root, exist_ok=True)
    for i in tqdm(range(ldm5.shape[0])):
        curr_lmd = ldm5[i]
        output_dir = os.path.join(output_root, os.path.basename(framename_list[i]).replace('png', 'txt')).replace('jpeg', 'txt')
        writeLandmarks(output_dir, curr_lmd)
