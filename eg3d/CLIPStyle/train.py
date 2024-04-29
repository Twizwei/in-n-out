'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    python CLIPStyle/train.py --size 512 \
                --batch 2 \
                --n_sample 2 \
                --output_dir NADA_ckpts/sketch \
                --lr 0.005 \
                --sample_truncation 1.0 \
                --frozen_gen_ckpt ./pretrained_models/ffhqrebalanced512-128.pkl \
                --iter 301 \
                --source_class "photo" \
                --target_class "sketch" \
                --auto_layer_k 14 \
                --auto_layer_iters 1 \
                --auto_layer_batch 2 \
                --output_interval 25 \
                --num_grid_outputs 1 \
                --clip_models "ViT-B/32" "ViT-B/16" \
                --clip_model_weights 1.0 1.0 \
                --mixing 0.0 \
                --save_interval 150 
'''

import argparse
import os
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np

import torch

from tqdm import tqdm

import shutil
import json
import pickle
import copy

import legacy
import dnnlib
from camera_utils import LookAtPoseSampler

from CLIPStyle.model.NADAGAN import NADAGAN, EG3DWrapper
from CLIPStyle.utils.file_utils import copytree, save_images, save_paper_image_grid
from CLIPStyle.utils.training_utils import mixing_noise
from CLIPStyle.options.train_options import TrainOptions

import pdb

#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True

def train(args):
    device = args.device
    # Set up networks, optimizers.
    print("Initializing networks...")
    # build up a pre-trained eg3d generator.
    with dnnlib.util.open_url(args.frozen_gen_ckpt) as f:
        eg3d_gen_frozen = legacy.load_network_pkl(f)['G_ema']
    with dnnlib.util.open_url(args.train_gen_ckpt) as f:
        eg3d_gen_train = legacy.load_network_pkl(f)['G_ema']
    
    eg3d_train = EG3DWrapper(eg3d_gen_train, device)
    eg3d_frozen = EG3DWrapper(eg3d_gen_frozen, device)
    net = NADAGAN(args, eg3d_frozen, eg3d_train, device)

    z_dim = eg3d_gen_train.z_dim

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    # torch.manual_seed(2)
    # np.random.seed(2)

    # Training loop
    fixed_z = torch.randn(args.n_sample, z_dim, device=device)
    pitch_range = 0.25
    yaw_range = 0.35
    with torch.no_grad():
        sample_yaw = (yaw_range * torch.randn(args.n_sample, 1)).to(device)
        sample_pitch = (pitch_range * torch.randn(args.n_sample, 1)).to(device)
        camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) 

        cam2world_pose = LookAtPoseSampler.sample(torch.pi/2 + sample_yaw, torch.pi/2 + sample_pitch, camera_lookat_point, radius=2.7, batch_size=args.n_sample, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device).unsqueeze(0).repeat(cam2world_pose.shape[0], 1, 1)
        fixed_c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    for i in tqdm(range(args.iter)):
        net.train()
        # sample z
        sample_z = mixing_noise(args.batch, z_dim, args.mixing, device)[0]

        # sample c
        sample_yaw = (yaw_range * torch.randn(args.batch, 1)).to(device)
        sample_pitch = (pitch_range * torch.randn(args.batch, 1)).to(device)
        camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) 

        cam2world_pose = LookAtPoseSampler.sample(torch.pi/2 + sample_yaw, torch.pi/2 + sample_pitch, camera_lookat_point, radius=2.7, batch_size=args.batch, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device).unsqueeze(0).repeat(cam2world_pose.shape[0], 1, 1)
        sample_c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        [sampled_src, sampled_dst], loss = net(sample_z, sample_c, input_is_latent=False, truncation_psi=args.sample_truncation)
        
        net.zero_grad()
        loss.backward()

        g_optim.step()

        tqdm.write(f"Iter: {i}, Clip loss: {loss}")

        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net(fixed_z, fixed_c, input_is_latent=False, truncation_psi=args.sample_truncation)

                if args.crop_for_cars:
                    sampled_dst = sampled_dst[:, :, 64:448, :]

                grid_rows = int(args.n_sample ** 0.5)

                if SAVE_SRC:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if SAVE_DST:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):

            # if args.sg3 or args.sgxl:

            #     snapshot_data = {'G_ema': copy.deepcopy(net.generator_trainable.generator).eval().requires_grad_(False).cpu()}
            #     snapshot_pkl = f'{ckpt_dir}/{str(i).zfill(6)}.pkl'

            #     with open(snapshot_pkl, 'wb') as f:
            #         pickle.dump(snapshot_data, f)

            # else:
            # torch.save(
            #     {
            #         "g_ema": net.generator_trainable.eg3d_gen.state_dict(),
            #         "g_optim": g_optim.state_dict(),
            #     },
            #     f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            # )
            snapshot_data = {'G_ema': copy.deepcopy(net.generator_trainable.eg3d_gen).eval().requires_grad_(False).cpu()}
            snapshot_pkl = f'{ckpt_dir}/{str(i).zfill(6)}.pkl'
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
            torch.save(
                {
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}_optim.pt",
            )

    torch.cuda.empty_cache()
    for i in range(args.num_grid_outputs):
        net.eval()

        with torch.no_grad():
            sample_z = mixing_noise(4, z_dim, 0, device)[0]
            sample_yaw = (yaw_range * torch.randn(4, 1)).to(device)
            sample_pitch = (pitch_range * torch.randn(4, 1)).to(device)
            camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) 

            cam2world_pose = LookAtPoseSampler.sample(torch.pi/2 + sample_yaw, torch.pi/2 + sample_pitch, camera_lookat_point, radius=2.7, batch_size=4, device=device)
            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device).unsqueeze(0).repeat(cam2world_pose.shape[0], 1, 1)
            sample_c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            [sampled_src, sampled_dst], _ = net(sample_z, sample_c, input_is_latent=False, truncation_psi=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]
        # save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.jpg")
        grid_rows = int(4 ** 0.5)
        save_images(sampled_dst, sample_dir, "dst_final", grid_rows, args.iter)
            

if __name__ == "__main__":
    args = TrainOptions().parse()

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    copytree("CLIPStyle/criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    shutil.copy2("CLIPStyle/model/NADAGAN.py", os.path.join(args.output_dir, "code", "NADAGAN.py"))
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    