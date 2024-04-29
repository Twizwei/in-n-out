import os
import glob
import argparse

import imageio
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_path", type=str, required=True, help="Path to target image")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the results")
    parser.add_argument("--img_format", type=str, default='png')
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--reverse_order", action='store_true')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--padding', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    video_out = imageio.get_writer(args.output_dir, mode='I', fps=args.fps, codec='libx264', quality=args.quality)
    if args.end_idx == None:
        frame_list = sorted(glob.glob(os.path.join(args.frames_path, f'*{args.img_format}')))[args.start_idx:]
    else:
        frame_list = sorted(glob.glob(os.path.join(args.frames_path, f'*{args.img_format}')))[args.start_idx:args.end_idx]
    if args.reverse_order:
        frame_list = sorted(frame_list, reverse=args.reverse_order)
    
    for frame_path in tqdm(frame_list):
        # if not args.padding:
        #     video_out.append_data(cv2.resize(imageio.imread(frame_path), (640, 640)))
        # else:
        #     frame_curr = cv2.imread(frame_path)
        #     frame_curr = cv2.resize(frame_curr, (1080, 1080))
        #     frame_curr = cv2.copyMakeBorder(frame_curr, 0, 0, int((1920-1080)/2), int((1920-1080)/2), cv2.BORDER_CONSTANT, value=(0,0,0))
        #     video_out.append_data(frame_curr[:, :, [2, 1, 0]])
        frame_curr = cv2.imread(frame_path)
        video_out.append_data(frame_curr[:, :, [2, 1, 0]])

