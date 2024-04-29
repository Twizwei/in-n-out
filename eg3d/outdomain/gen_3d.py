# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""(Not fully clean)
This code can generate interp video given a latent code.
Example:
python outdomain/gen_3d.py --outdir=/fs/nexus-projects/video3dgan/inversion_results/Halloween4/outdomain_iters20000_1105/train_sr/shape --trunc=1.0 --seeds=0 --grid=1x1 --ckpt_path /fs/nexus-projects/video3dgan/inversion_results/Halloween4/outdomain_iters20000_1105/train_sr/triplanes.pt --network=pretrained_models/ffhqrebalanced512-128.pkl --latents_path /fs/nexus-projects/video3dgan/inversion_results/Halloween4/latents.pt --w_type=w_plus --shapes=False --interpolate=true --frame_idx 0
"""

import os
import sys
sys.path.append(".")
sys.path.append("..")
import re
from typing import List, Optional, Tuple, Union
import pdb
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import scipy.io
import cv2

import torch
from tqdm import tqdm
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler, compute_rotation
from torch_utils import misc
from outdomain.triplanes_split import ComposeTriplane

#----------------------------------------------------------------------------
def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0,0] = 2985.29/700
    intrinsics[1,1] = 2985.29/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    return intrinsics

def fix_pose_orig(pose):
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3]/radius * 2.7
    return pose

def angle2matrix(angle, trans, device, src_dir='/fs/nexus-scratch/yiranx/codes/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/rednose2/epoch_20_000000/frame0055.mat'):    
    R = compute_rotation(torch.from_numpy(angle), device='cpu')[0].numpy()
    trans[2] += -10
    c = -np.dot(R, trans)
    pose = np.eye(4)
    pose[:3, :3] = R

    c *= 0.27 # normalize camera radius
    c[1] += 0.006 # additional offset used in submission
    c[2] += 0.161 # additional offset used in submission
    pose[0,3] = c[0]
    pose[1,3] = c[1]
    pose[2,3] = c[2]

    focal = 2985.29 # = 1015*1024/224*(300/466.285)#
    pp = 512#112
    w = 1024#224
    h = 1024#224

    count = 0
    K = np.eye(3)
    K[0][0] = focal
    K[1][1] = focal
    K[0][2] = w/2.0
    K[1][2] = h/2.0
    K = K.tolist()

    Rot = np.eye(3)
    Rot[0, 0] = 1
    Rot[1, 1] = -1
    Rot[2, 2] = -1        
    pose[:3, :3] = np.dot(pose[:3, :3], Rot)

    pose = pose.tolist()
    # out = {}
    # out["intrinsics"] = K
    # out["pose"] = pose

    pose = torch.from_numpy(fix_pose_orig(pose)).to(device)
    intrinsics = torch.from_numpy(fix_intrinsics(K))
    # label = torch.from_numpy(np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)])).to(device)

    return pose, intrinsics


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

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

def gen_interp_video(compose_triplanes, phi, mp4: str, seeds, latents_path, f_idx, w_type='cano', padding=False, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, save_frames=False, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(compose_triplanes.G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    # ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    latents = torch.load(latents_path)
    if isinstance(latents, dict):
        ws = latents['w_plus'][f_idx:f_idx+1].detach().to(device)
    elif isinstance(latents, torch.Tensor):
        ws = latents[f_idx:f_idx+1].detach().to(device)
    phi_curr = phi[f_idx:f_idx+1].to(device)
    # _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000//2
    voxel_resolution = 512
    if not gen_shapes:
        video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
        if save_frames:
            os.makedirs(mp4[:-4], exist_ok=True)

    if gen_shapes:
        # outdir = 'interpolation_{}_{}/'.format(all_seeds[0], all_seeds[1])
        outdir = os.path.dirname(mp4)
        os.makedirs(outdir, exist_ok=True)
    all_poses = []
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    # src_dir=os.path.join('/fs/nexus-scratch/yiranx/codes/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results', os.path.dirname(latents_path).split('/')[-1], 'epoch_20_000000/frame{0:04d}.mat'.format(f_idx))
    # dict_load = scipy.io.loadmat(src_dir)
    # angle = dict_load['angle']
    # trans = dict_load['trans'][0]
    # camera_lookat_point = torch.tensor([0.0000, 0.0000, 0.0000]).to(device)
    with torch.no_grad():
        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    # pitch_range = 0.25
                    # yaw_range = 0.25
                    pitch_range = 0.15
                    yaw_range = 0.15
                    
                    # normal spiral
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)

                    # pdb.set_trace()
                    # spiral from a frame pose
                    # angle_curr = [[angle[:,0].item() + yaw_range * np.sin(4 * 3.14 * frame_idx / (num_keyframes * w_frames)), angle[:,1].item() - pitch_range + pitch_range * np.cos(4 * 3.14 * frame_idx / (num_keyframes * w_frames)), angle[:,2].item() ]]
                    # angle_curr = np.array(angle_curr)
                    # cam2world_pose = angle2matrix(angle_curr, trans, device=device)[0].to(torch.float32)
                    
                    all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                    # cam2world_pose = all_poses[frame_idx].to(device)
                    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                    
                    entangle = 'camera'
                    out = compose_triplanes(w.unsqueeze(0), c[0:1], phi_curr)

                    img = out['image_full'][0]
                    # if entangle == 'conditioning':
                    #     c_forward = torch.cat([LookAtPoseSampler.sample(3.14/2,
                    #                                                     3.14/2,
                    #                                                     camera_lookat_point,
                    #                                                     radius=2.7, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    #     w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    #     img = G.synthesis(ws=w_c, c=c_forward, noise_mode='const')[image_mode][0]
                    # elif entangle == 'camera':
                    #     img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]
                    #     # img = G.synthesis(ws=ws, c=c[0:1], noise_mode='const')[image_mode][0]
                    # elif entangle == 'both':
                    #     w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    #     img = G.synthesis(ws=w_c, c=c[0:1], noise_mode='const')[image_mode][0]

                    if image_mode == 'image_depth':
                        img = -img
                        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                    if not padding:
                        imgs.append(img)
                    else:
                        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # [-1, 1] -> [0, 255]
                        img = img.permute(1, 2, 0).cpu().numpy()
                        img = cv2.resize(img, (1080, 1080))
                        img = cv2.copyMakeBorder(img, 0, 0, int((1920-1080)/2), int((1920-1080)/2), cv2.BORDER_CONSTANT, value=(0,0,0))
                        img = torch.from_numpy(img).permute(2, 0, 1)
                        imgs.append(img)
                    
                    if gen_shapes:
                        if not os.path.exists(os.path.join(os.path.dirname(mp4), 'shape.npy')):
                            # generate shapes
                            # print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))
                            
                            samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=compose_triplanes.G.rendering_kwargs['box_warp'])
                            samples = samples.to(device)
                            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                            transformed_ray_directions_expanded[..., -1] = -1

                            head = 0
                            
                            with tqdm(total = samples.shape[1]) as pbar:
                                    while head < samples.shape[1]:
                                        torch.manual_seed(0)
                                        # sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                        # Or dive into function...
                                        # ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
                                        ws = ws.squeeze().unsqueeze(0)
                                        planes_f = compose_triplanes.G.backbone.synthesis(ws, noise_mode='const')
                                        planes_f = planes_f.view(len(planes_f), 3, 32, planes_f.shape[-2], planes_f.shape[-1])
                                        sigma_f = compose_triplanes.G.renderer.run_model(planes_f, compose_triplanes.G.decoder, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], compose_triplanes.G.rendering_kwargs)['sigma']
                                        
                                        planes_o = compose_triplanes.new_triplanes.triplanes
                                        out = compose_triplanes.new_triplanes.renderer.run_model_extend(phi_curr, planes_o, compose_triplanes.new_triplanes.decoder, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], compose_triplanes.G.rendering_kwargs)
                                        sigma_o = out['sigma']
                                        blendings = out['blending']

                                        sigma = sigma_o * blendings + sigma_f * (1 - blendings)
                                        
                                        # if we only need in-distribution part
                                        # sigma = sigma_f

                                        sigmas[:, head:head+max_batch] = sigma
                                        head += max_batch
                                        pbar.update(max_batch)

                            sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                            sigmas = np.flip(sigmas, 0)
                            
                            pad = int(30 * voxel_resolution / 256)
                            pad_top = int(38 * voxel_resolution / 256)
                            sigmas[:pad] = 0
                            sigmas[-pad:] = 0
                            sigmas[:, :pad] = 0
                            sigmas[:, -pad_top:] = 0
                            sigmas[:, :, :pad] = 0
                            sigmas[:, :, -pad:] = 0

                            # output_ply = False
                            # if output_ply:
                            #     from shape_utils import convert_sdf_samples_to_ply
                            #     convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                            # else: # output mrc
                            #     with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                            #         mrc.data[:] = sigmas
                            np.save(os.path.join(os.path.dirname(mp4), 'shape.npy'), sigmas)

            if gen_shapes:
                break                
            if not padding:
                frame = layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h)
                video_out.append_data(frame)
            else:
                frame = layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h, float_to_uint8=False)
                video_out.append_data(frame)
            if save_frames:
                imageio.imwrite(os.path.join(mp4[:-4], f'{frame_idx:04d}.png'), frame)
    if not gen_shapes:
        video_out.close()
    all_poses = np.stack(all_poses)

    # if gen_shapes:
    #     print(all_poses.shape)
    #     with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
    #         np.save(f, all_poses)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
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

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--ckpt_path', help='checkpoint path', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=60)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)
@click.option('--latents_path', type=str, help='path to the latent codes', required=True)
@click.option('--w_type', type=str, help='type of latent code')
@click.option('--frame_idx', type=int, help='frame index in a video', default=0)
@click.option('--padding', type=bool, help='padding or not')
@click.option('--save_frames', type=bool, default=False, help='save frames or not')



def generate_images(
    network_pkl: str,
    ckpt_path: str, 
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    outdir: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
    latents_path: str, 
    w_type: Optional[str],
    frame_idx: Optional[int],
    padding: Optional[bool],
    save_frames: Optional[bool],
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    if network_pkl.endswith("pkl"):
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f, g_ema_only=True)['G_ema'].to(device) # type: ignore
    elif network_pkl.endswith("pt"):
        G = torch.load(network_pkl)['G_ema'].to(device)


    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr
    
    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    # build composed triplanes for the out-of-distribution item
    compose_triplanes = ComposeTriplane(G).to(device)

    # load ckpt
    ckpt = torch.load(ckpt_path)
    compose_triplanes.load_state_dict(ckpt['compose_triplanes'])
    phi = ckpt['phi']
    if interpolate:
        if w_type == 'cano':
            output = os.path.join(outdir, 'rgb_interpolation_cano.mp4')
        elif w_type == 'w_plus':
            output = os.path.join(outdir, 'rgb_interpolation_frame{0:05d}.mp4'.format(frame_idx))
        gen_interp_video(compose_triplanes, phi, mp4=output, latents_path=latents_path, f_idx=frame_idx, padding=padding, w_type=w_type, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, save_frames=save_frames, device=device)
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
