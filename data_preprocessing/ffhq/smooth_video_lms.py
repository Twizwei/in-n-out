"""
Using a smoothing window to get smoother facial landmarks.
Example:
python smooth_video_lms.py -f /fs/nexus-scratch/yiranx/data/frames/rednose2/detections 
"""
import argparse
import numpy as np
from tqdm import tqdm
import os
import glob
from collections import deque

import pdb

def readLandmarks(txt_path):
    return np.loadtxt(txt_path)

def writeLandmarks(txt_path, landmarks, fmt='%1.1f'):
    return np.savetxt(txt_path, landmarks, fmt)

def main(args):
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1  # window length

    input_path = args.video_fp
    reader = sorted(glob.glob(os.path.join(input_path, '*txt')))  # we consider txt at this time

    queue_ver = deque()
    final_ver = []
    pre_ver = None
    
    for i, file_path in enumerate(tqdm(reader)):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i > args.end:
            break
        
        if i == 0:
            ver = readLandmarks(file_path)
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())
        else:
            ver = readLandmarks(file_path)
            queue_ver.append(ver.copy())
        
        pre_ver = ver
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            final_ver.append(ver_ave)
            queue_ver.popleft()
    # pad last few frames
    for _ in range(n_next):
        queue_ver.append(ver.copy())
        ver_ave = np.mean(queue_ver, axis=0)
        final_ver.append(ver_ave)
        queue_ver.popleft()
    assert len(final_ver) == len(reader), "The number of landmarks does not match the number of video frames!"

    # save smoothed results
    # output_root = input_path.replace('detections', 'smoothed_detections')
    output_root = input_path
    os.makedirs(output_root, exist_ok=True)

    print("Saving results...")
    for i, file_path in enumerate(tqdm(reader)):
        # output_dir = file_path.replace('detections', 'smoothed_detections')
        output_dir = file_path
        curr_ver = final_ver[i]
        writeLandmarks(output_dir, curr_ver)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time smooth video.')
    parser.add_argument('-f', '--video_fp', type=str, help='')
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
    parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')

    args = parser.parse_args()
    main(args)