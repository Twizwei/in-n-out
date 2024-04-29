# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import sys
sys.path.append(".")
sys.path.append("..")
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import pdb

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    
    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)
    
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)  # almost the same as torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        imgs = []
        angle_p = -0.2
        # for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
        for angle_y, angle_p in [(0, angle_p), ]:
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            # pdb.set_trace()
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            
            # ws = torch.load('/home/yiranx/codes/eg3d/eg3d/out/optim_multi_dynamic/11986_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.2res/latents.pt')['w_plus_cano'].to(device)
            # img = G.synthesis(ws, camera_params)['image']

            # synthesis
            cam2world_matrix = camera_params[:, :16].view(-1, 4, 4)
            intrinsics = camera_params[:, 16:25].view(-1, 3, 3)
            neural_rendering_resolution = G.neural_rendering_resolution
            # Create a batch of rays for volume rendering
            ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)  # [N, 128*128, 3], [N, 128*128, 3]

            # Create triplanes by running StyleGAN backbone
            N, M, _ = ray_origins.shape 
            planes = G.backbone.synthesis(ws, update_emas=False,)  # [N, 96, 256, 256]
            # Reshape output into three 32-channel planes
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])  # [N, 3, 32, 256, 256]
            pdb.set_trace()
            rendering_options = G.rendering_kwargs
            # Perform volume rendering, color/feature [B,128*128,32], density/opacity [B,128*128,1], weights [B,128*128,1]
            G.renderer.plane_axes = G.renderer.plane_axes.to(ray_origins.device)

            if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
                ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
                is_ray_valid = ray_end > ray_start
                if torch.any(is_ray_valid).item():
                    ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                    ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
                depths_coarse = G.renderer.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            else:
                # Create stratified depth samples
                depths_coarse = G.renderer.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            
            batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape  # [1, 16384, 48, 1]

            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

            pdb.set_trace()
            out = G.renderer.run_model(planes, G.decoder, sample_coordinates, sample_directions, rendering_options)
            colors_coarse = out['rgb']
            densities_coarse = out['sigma']
            colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
            densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

            # Fine Pass
            N_importance = rendering_options['depth_resolution_importance']
            if N_importance > 0:
                _, _, weights = G.renderer.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

                depths_fine = G.renderer.sample_importance(depths_coarse, weights, N_importance)

                sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
                sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

                out = G.renderer.run_model(planes, G.decoder, sample_coordinates, sample_directions, rendering_options)
                colors_fine = out['rgb']
                densities_fine = out['sigma']
                colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
                densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
                # pdb.set_trace()
                # seed_1_render_vars = torch.load('/home/yiranx/codes/eg3d/eg3d/out/render_vars_seed_00001.pt')
                # colors_coarse = colors_coarse + seed_1_render_vars['rgb_coarse'].to(device)
                # densities_coarse = densities_coarse + seed_1_render_vars['density_coarse'].to(device)
                # colors_fine = colors_fine + seed_1_render_vars['rgb_fine'].to(device)
                # densities_fine = densities_fine + seed_1_render_vars['density_fine'].to(device)
                all_depths, all_colors, all_densities = G.renderer.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                    depths_fine, colors_fine, densities_fine)
                pdb.set_trace()
                # Aggregate
                # rgb_final, depth_final, weights = G.renderer.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
                deltas = all_depths[:, :, 1:] - all_depths[:, :, :-1]  # [N, 16384, 95, 1]
                colors_mid = (all_colors[:, :, :-1] + all_colors[:, :, 1:]) / 2  # [N, 16384, 95, 1]
                densities_mid = (all_densities[:, :, :-1] + all_densities[:, :, 1:]) / 2  # [N, 16384, 95, 1]
                depths_mid = (all_depths[:, :, :-1] + all_depths[:, :, 1:]) / 2  # [N, 16384, 95, 1]


                if rendering_options['clamp_mode'] == 'softplus':
                    import torch.nn.functional as F
                    densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
                else:
                    assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

                density_delta = densities_mid * deltas  # [N, 16384, 95, 1]

                alpha = 1 - torch.exp(-density_delta)  # [N, 16384, 95, 1]

                alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)  # [N, 16384, 96, 1]
                weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]  # [N, 16384, 95, 1]

                composite_rgb = torch.sum(weights * colors_mid, -2)
                weight_total = weights.sum(2)
                composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

                # clip the composite to min/max range of depths
                composite_depth = torch.nan_to_num(composite_depth, float('inf'))
                depth_final = torch.clamp(composite_depth, torch.min(all_depths), torch.max(all_depths))

                if rendering_options.get('white_back', False):
                    composite_rgb = composite_rgb + 1 - weight_total
                pdb.set_trace()
                rgb_final = composite_rgb * 2 - 1 # Scale to (-1, 1)
            else:
                rgb_final, depth_final, weights = G.renderer.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            feature_samples, depth_samples, weights_samples = rgb_final, depth_final, weights.sum(2)

            # Reshape into 'raw' neural-rendered image
            H = W = G.neural_rendering_resolution
            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()  # [B,32,128,128]
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)  # [B,1,128,128]

            # Run superresolution to get final image
            rgb_image = feature_image[:, :3]  # raw image
            sr_image = G.superresolution(rgb_image, feature_image, ws, noise_mode=G.rendering_kwargs['superresolution_noise_mode'],)

            img = sr_image
            pdb.set_trace()
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

        img = torch.cat(imgs, dim=2)

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        # get sdf sigma
                        # sigma = G.sample(samples[:, head:head+max_batch], 
                                        # transformed_ray_directions_expanded[:, :samples.shape[1]-head], 
                                        # z, conditioning_params, 
                                        # truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, 
                                        # noise_mode='const')['sigma']
                        # Or dive into function...
                        # ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
                        planes = G.backbone.synthesis(ws, noise_mode='const')
                        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
                        sigma = G.renderer.run_model(planes, G.decoder, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], G.rendering_kwargs)['sigma']
                        
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
