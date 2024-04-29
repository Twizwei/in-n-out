"""
Simply use mask and extra triplane to do alpha/mask blending inference.
"""

import os
import sys
sys.path.append(".")
sys.path.append("..")

import legacy
import dnnlib
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, angles2mat
from outdomain.od_dataloader import MutilFrameDatasetDynamic as MutilFrameDataset
from outdomain.triplanes_split import TriplaneRadianceField

import click
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import imageio
import numpy as np
import PIL.Image
import lpips
from tqdm import tqdm

import pdb


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--triplane_ckpt', help='Triplan ckpt', required=True)
@click.option('--target_path', help='Target frame path', type=str, required=True, metavar='DIR')
@click.option('--latents_path', help='latent code path', type=str, )
@click.option('--outdir', help='Where to save the output results', type=str, required=True, metavar='DIR')
@click.option('--batch_size', type=int, help='batch size', default=1)
@click.option('--use_mask', type=bool, help='if use mask', default=False, show_default=True)
@click.option('--output_resolution', type=int, help='neural rendering output resolution', default=128, show_default=True)


def outdomain_inv(
        network_pkl,
        triplane_ckpt,
        target_path,
        latents_path,
        outdir,
        batch_size=1,
        use_mask=True,
        output_resolution=128,
    ):
    device = torch.device('cuda')
    # load a generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # load initial latent codes (from other inversion method)
    latents = torch.load(latents_path)
    if isinstance(latents, dict):
        w_plus = latents['w_plus']
    elif isinstance(latents, torch.Tensor):
        w_plus = latents
    
    # randomly initialize phi codes
    phi = torch.randn(w_plus.shape[0], 512).clone().to(device).requires_grad_(True)

    # build dataset and dataloader
    video_dataset = MutilFrameDataset(data_root=target_path, yaw_opt=None, pitch_opt=None, use_pre_pose=True)
    video_dataset.w_plus = w_plus
    video_dataset.phi = phi
    # Will return img_tensor, mask_tensor, cam_param, w_plus_opt_curr, img_path
    train_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # build triplanes for the out-of-distribution item
    rendering_kwargs = {'depth_resolution': 48, 
                        'depth_resolution_importance': 48, 
                        'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 
                        'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2], 
                        'image_resolution': 512, 'disparity_space_sampling': False, 
                        'clamp_mode': 'softplus', 
                        'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC', 
                        'c_gen_conditioning_zero': False, 'gpc_reg_prob': 0.8, 'c_scale': 1.0, 
                        'superresolution_noise_mode': 'none', 'density_reg': 0.25, 
                        'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 
                        'sr_antialias': True
                    }
    triplaneField = TriplaneRadianceField(
        rendering_kwargs,
        num_triplanes=1,
        num_channels=32,
        plane_resolution=256,
        neural_rendering_resolution=output_resolution,
        init_way='normal',
    ).to(device)

    triplaneField.load_state_dict(torch.load(triplane_ckpt)['triplanes'])
       
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    
    # visualize and save results
    os.makedirs(os.path.join(outdir, 'frames', 'projected_full'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_o'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'targets'), exist_ok=True)
    eval_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Saving results...")
    f = open(os.path.join(outdir, 'target_frames.txt'), 'w')    
    triplaneField.eval()
    G.eval()
    with torch.no_grad():
        for itr, data in enumerate(eval_dataloader):
            target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, img_path = data
            camera_params = camera_params.to(device)
            w_plus_curr = w_plus_curr.to(device)
            phi_curr = phi_curr.to(device)
            masks_tensor = F.interpolate(masks_tensor, size=(triplaneField.neural_rendering_resolution, triplaneField.neural_rendering_resolution), mode='nearest')
            out = triplaneField(c=camera_params)
            
           
            target_images = (target_images + 1) * (255/2)
            target_images = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_o = out['image_raw']
            synth_image_o = (synth_image_o + 1) * (255/2)
            synth_image_o = synth_image_o.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_f = G.synthesis(w_plus_curr, camera_params)['image_raw']
            synth_image_f = (synth_image_f + 1) * (255/2)
            synth_image_f = synth_image_f.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            masks_np = masks_tensor.permute(0, 2, 3, 1).clamp(0, 1)[0].cpu().numpy()

            synth_image_full = masks_np * synth_image_f + (1 - masks_np) * synth_image_o

            synth_image_full = np.clip(synth_image_full, 0, 255).astype(np.uint8)
                

            PIL.Image.fromarray(synth_image_o, 'RGB').save(os.path.join(outdir, 'frames', 'projected_o', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_full, 'RGB').save(os.path.join(outdir, 'frames', 'projected_full', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(target_images, 'RGB').save(os.path.join(outdir, 'frames', 'targets', 'target_{0:05d}.png'.format(itr)))
            f.write(os.path.basename(img_path[0]) + '\n')

    f.close()

if __name__ == '__main__':
    outdomain_inv()