"""
Project given image to the latent space of pretrained network pickle.
Optimize w_renderer for multiple frames

"""

import os
import sys
sys.path.append(".")
sys.path.append("..")

import legacy
import dnnlib
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, angles2mat
from outdomain.od_dataloader import MutilFrameDatasetDynamic as MutilFrameDataset
from outdomain.triplanes_split import ComposeTriplane

import click
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import imageio
import numpy as np
import PIL.Image
from PIL import Image   
import lpips
from tqdm import tqdm

import pdb

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def smooth_var(var):
    var_p = var[2:-2] + 0.75 * (var[1:-3] + var[3:-1]) + 0.25 * (var[:-4] + var[4:])
    var_p = var_p / 3
    # pad the first and last two frames
    var_p = torch.cat([var[:2], var_p, var[-2:]], dim=0)
    return var_p

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--ckpt_path', help='checkpoint path', required=True)
@click.option('--target_path', help='Target frame path', type=str, required=True, metavar='DIR')
@click.option('--latents_path', help='latent code path', type=str, )
@click.option('--outdir', help='Where to save the output results', type=str, required=True, metavar='DIR')
@click.option('--batch_size', type=int, help='batch size', default=1)
@click.option('--remove_ood', type=bool, help='if we want to remove ood object', default=False)
@click.option('--smooth_out', type=bool, help='if smooth the results', default=False)
@click.option('--kernel_size', type=int, help='kernel size for smoothing', default=3)


def outdomain_inv(
        network_pkl,
        ckpt_path,
        target_path,
        latents_path,
        outdir,
        batch_size=1,
        remove_ood=False,
        smooth_out=False,
        kernel_size=5,
    ):
    # load generator
    device = torch.device('cuda')
    # load a generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # build composed triplanes for the out-of-distribution item
    compose_triplanes = ComposeTriplane(G).to(device)

    # load ckpt
    ckpt = torch.load(ckpt_path)
    compose_triplanes.load_state_dict(ckpt['compose_triplanes'])
    phi = ckpt['phi']

    # load latent codes
    latents = torch.load(latents_path)
    # latents = ckpt['w_plus']
    if isinstance(latents, dict):
        w_plus = latents['w_plus']
    elif isinstance(latents, torch.Tensor):
        w_plus = latents

    # build dataset and dataloader
    video_dataset = MutilFrameDataset(data_root=target_path, yaw_opt=None, pitch_opt=None, use_pre_pose=True)
    
    # if smooth_out:
    #     w_plus = smooth_var(w_plus)
    #     phi = smooth_var(phi)
    
    video_dataset.w_plus = w_plus
    video_dataset.phi = phi

   
    # visualize and save results
    os.makedirs(os.path.join(outdir, 'frames', 'projected'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_o'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_f'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_sr'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_o_sr'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'targets'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'depth'), exist_ok=True)
    eval_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Saving results...")
    print(outdir)
    f = open(os.path.join(outdir, 'target_frames.txt'), 'w')
    compose_triplanes.eval()
    
    # smooth_kernel = 1/3 * torch.tensor([0.25, 0.75, 1, 0.75, 0.5]).view(5, 1).to(device)
    # smooth_kernel = 1/(kernel_size - 2) * torch.tensor([0.25, 0.75, 1, 0.75, 0.25]).view(kernel_size, 1).to(device)
    # smooth_kernel = torch.tensor([0.05, 0.25, 0.75, 0.95, 1]).view(kernel_size, 1).to(device)
    smooth_kernel = torch.tensor([0.25, 0.75, 1]).view(kernel_size, 1).to(device)
    # smooth_kernel = torch.tensor([0.25, 0.5, 0.75, 1]).view(kernel_size, 1).to(device)
    smooth_kernel = smooth_kernel / torch.sum(smooth_kernel)
    acc_w_plus = []
    acc_phi = []
    acc_sr_rgb = []
    final_w_plus = []
    final_phi = []
    final_sr_rgb = []

    with torch.no_grad():
        compose_triplanes.acc_toRGB_input = []
        compose_triplanes.acc_rgb = []
        for itr, data in enumerate(tqdm(eval_dataloader)):
            
            if video_dataset.depth_maps is not None:
                target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, _, img_path = data
            else:
                target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, img_path = data
            camera_params = camera_params.to(device)
            w_plus_curr = w_plus_curr.to(device)
            phi_curr = phi_curr.to(device)
            
            acc_w_plus.append(w_plus_curr)
            acc_phi.append(phi_curr)
            
            if smooth_out and len(acc_w_plus) >= kernel_size and len(acc_phi) >= kernel_size:
                w_plus_curr = torch.sum(smooth_kernel.view(kernel_size, 1, 1) * torch.cat(acc_w_plus, dim=0), dim=0, keepdim=True)
                phi_curr = torch.sum(smooth_kernel * torch.cat(acc_phi, dim=0), dim=0, keepdim=True)
                acc_w_plus.pop(0)
                acc_phi.pop(0)
                final_w_plus.append(w_plus_curr)
                final_phi.append(phi_curr)
                
            else:
                final_w_plus.append(w_plus_curr)
                final_phi.append(phi_curr)

            out = compose_triplanes(w_plus_curr, camera_params, phi_curr, masks_tensor, smooth_kernel=smooth_kernel, remove_ood=remove_ood, smooth_toRGB=smooth_out)

            synth_image = out['image_raw_full']
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            target_images = (target_images + 1) * (255/2)
            target_images = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_o = out['image_raw_o']
            synth_image_o = (synth_image_o + 1) * (255/2)
            synth_image_o = synth_image_o.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_f = compose_triplanes.G.synthesis(w_plus_curr, camera_params)['image_raw']
            synth_image_f = (synth_image_f + 1) * (255/2)
            synth_image_f = synth_image_f.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_full_sr = out['image_full']
            acc_sr_rgb.append(synth_image_full_sr)
            # if smooth_out and len(acc_sr_rgb) >= kernel_size:
            #     synth_image_full_sr = torch.sum(smooth_kernel.view(kernel_size, 1, 1, 1) * torch.cat(acc_sr_rgb, dim=0), dim=0, keepdim=True)
            #     acc_sr_rgb.pop(0)
            #     final_sr_rgb.append(synth_image_full_sr)
            # else:
            #     final_sr_rgb.append(synth_image_full_sr)
            synth_image_full_sr = (synth_image_full_sr + 1) * (255/2)
            synth_image_full_sr = synth_image_full_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_o_sr = out['image_o']
            synth_image_o_sr = (synth_image_o_sr + 1) * (255/2)
            synth_image_o_sr = synth_image_o_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            depth_map = out['image_depth_full']
            depth_map = render_depth(depth_map.squeeze().cpu())
            depth_map.save(os.path.join(outdir, 'frames', 'depth', 'depth_{0:05d}.png'.format(itr)))
                
            PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'frames', 'projected', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_o, 'RGB').save(os.path.join(outdir, 'frames', 'projected_o', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_f, 'RGB').save(os.path.join(outdir, 'frames', 'projected_f', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_full_sr, 'RGB').save(os.path.join(outdir, 'frames', 'projected_sr', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_o_sr, 'RGB').save(os.path.join(outdir, 'frames', 'projected_o_sr', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(target_images, 'RGB').save(os.path.join(outdir, 'frames', 'targets', 'target_{0:05d}.png'.format(itr)))
            f.write(os.path.basename(img_path[0]) + '\n')

    f.close()


if __name__ == '__main__':
    torch.manual_seed(20221026)
    torch.cuda.manual_seed(20221026)
    np.random.seed(20221026)
    outdomain_inv()
