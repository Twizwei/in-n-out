"""
Optimize for yaw and pitch given a pose matrix.
"""
import os
import argparse
import json
import torch
import numpy as np
from camera_utils import LookAtPoseSampler
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_param_path', type=str, required=True, help='path to cropped/aligned frames')
    parser.add_argument('--idx', type=int, required=True, default=55, help='path to the original (uncropped) frames')
    parser.add_argument('--output_dir', type=str, required=True, help='output path.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--iters', type=int, default=100, help='optimization iterations')
    args = parser.parse_args()

    device = 'cuda'

    yaw_opt = torch.zeros(1, 1)
    pitch_opt = torch.zeros(1, 1)

    yaw_opt = yaw_opt.clone().to(device).requires_grad_(True)
    pitch_opt = pitch_opt.clone().to(device).requires_grad_(True)


    optimizer = torch.optim.Adam([
                                    {'params': [yaw_opt], 'lr': float(args.lr)},
                                    {'params': [pitch_opt], 'lr': float(args.lr)},
                                    ])

    with open(args.cam_param_path, 'r') as f:
        cam_params = json.load(f)['labels']
    
    cam_param = torch.tensor(cam_params[args.idx][1])[:16].reshape(-1, 4, 4).to(device)
    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)

    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    for iter in range(args.iters):
        # Learning rate schedule.
        t = iter / args.iters
        lr_ramp = min(1.0, (1.0 - t) / 0.25)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / 0.05)
        lr = args.lr * lr_ramp
        
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_opt,
                                                        3.14/2 -0.05 + pitch_opt,
                                                        camera_lookat_point, radius=2.7, device=device)


        loss = criterion(cam2world_pose, cam_param)
        loss.backward()
        optimizer.step()
        print("Iter: {}/{}, Loss:{} ".format(iter, args.iters, loss.item()) )
    print("Yaw: {} , Pitch: {}".format(yaw_opt, pitch_opt))
    torch.save({'yaw': yaw_opt.cpu(), 'pitch':pitch_opt.cpu()}, args.output_dir)
    print(cam2world_pose)
    print(cam_param)

    