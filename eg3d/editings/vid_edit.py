"""
Example:
python vid_edit.py --results_path ../out/optim_multi_dynamic/mask1_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/interfacegan/eyeglasses.mp4 --latents_path ../out/optim_multi_dynamic/mask1_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/latents.pt --camera_params_path ../out/optim_multi_dynamic/mask1_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/cam_params.pt --saved_directions_path /fs/nexus-scratch/yiranx/codes/eg3d/eg3d/interfacegan/out/boundaries/eyeglasses/boundary.npy --batch_size 2

python vid_edit.py --results_path ../out/optim_multi_dynamic/rednose2_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/interfacegan/eyeglasses_pti.mp4 --latents_path ../out/optim_multi_dynamic/rednose2_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/latents.pt --camera_params_path ../out/optim_multi_dynamic/rednose2_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/cam_params.pt --saved_directions_path /fs/nexus-scratch/yiranx/codes/eg3d/eg3d/interfacegan/out/boundaries/eyeglasses/boundary.npy --batch_size 2  --network_pkl /fs/nexus-scratch/yiranx/codes/eg3d/eg3d/out/optim_multi_dynamic/rednose2_full_1.0lpips_1.0l2_lr1e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/pti_test/model_pti_refine.pkl
"""

import os
import sys
sys.path.append(".")
sys.path.append("..")
import argparse

import imageio
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from latent_editor import LatentEditor
import legacy
import dnnlib
from camera_utils import LookAtPoseSampler
from CLIPStyle.mapper.datasets.latents_dataset import LatentsDataset

import pdb

if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True, help="Where to save the results")
    parser.add_argument("--latents_path", type=str, required=True, help="Where to get the latent codes.")
    parser.add_argument("--camera_params_path", type=str, default=None, help="Where to get camera parameters.")
    parser.add_argument("--saved_directions_path", type=str, required=True, help="Path to the saved direction file.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for video rendering.")
    parser.add_argument("--network_pkl", type=str, default='../pretrained_models/ffhqrebalanced512-128.pkl', help='Path to the pretrained checkpoint of G')
    parser.add_argument("--encode_space", type=str, default='WP', help="Path to the latents")
    parser.add_argument("--edit_factor", type=float, default=2.2, help="How much to edit an image")
    parser.add_argument("--output_latents", action='store_true')

    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    # load a generator
    print("Loading G from {}".format(args.network_pkl))
    with dnnlib.util.open_url(args.network_pkl) as f:
        G = legacy.load_network_pkl(f, g_ema_only=True)['G_ema'].to(device) # type: ignore

    # wrap G into a editor
    editor = LatentEditor(G)

    # load latents and cam parameters
    latents_old = torch.load(args.latents_path)['w_plus'].detach()
    if latents_old.shape[1] == 1:
        latents_old = latents_old.repeat(1, 14, 1)
    cam_params = torch.load(args.camera_params_path)
    latent_dataset = LatentsDataset(latents_old, opts=None, cam_params=cam_params)
    latent_dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # load a pretrained interfacegan direction: --saved_directions_path
    direction = torch.from_numpy(np.load(args.saved_directions_path)).to(device)

    # video_out = imageio.get_writer(args.results_path, mode='I', fps=30, codec='libx264')
    frame_folder = os.path.join(os.path.dirname(args.results_path), 'frames')
    os.makedirs(frame_folder, exist_ok=True)
    all_latents = []
    global_iter = 0
    for data in tqdm(latent_dataloader):
        ws, cam = data
        ws = ws.to(device)
        cam = cam.to(device)
        # import pdb; pdb.set_trace()
        images, latents = editor.apply_interfacegan(ws, cam, direction, factor=args.edit_factor, factor_range=None, output_latents=args.output_latents, space=args.encode_space, if_stack=False)
        # if images.shape[0] == args.batch_size:
        #     for batch_idx in range(args.batch_size):
        #         video_out.append_data(images[batch_idx])  
        #         all_latents.append(latents[batch_idx])
        #         imageio.imwrite(os.path.join(frame_folder, 'frame_{0:05d}.png'.format(global_iter)), images[batch_idx])
        #         global_iter +=1
        # else:
        #     video_out.append_data(images[0]) 
        #     all_latents.append(latents[0])
        #     global_iter +=1
        for batch_idx in range(images.shape[0]):
            # video_out.append_data(images[batch_idx])  
            all_latents.append(latents[batch_idx])
            imageio.imwrite(os.path.join(frame_folder, 'frame_{0:05d}.png'.format(global_iter)), images[batch_idx])
            global_iter +=1

    if args.output_latents:
        all_latents = torch.stack(all_latents).cpu()
        torch.save({'w_plus': all_latents}, os.path.join(os.path.dirname(args.results_path), 'latents.pt'))
        