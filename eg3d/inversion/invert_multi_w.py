"""
Project given image to the latent space of pretrained network pickle.
Optimize w_renderer for multiple frames
Example:
python inversion/invert_multi_ws.py --network=pretrained_models/ffhqrebalanced512-128.pkl --target_path /home/yiranx/sensei-fs-symlink/users/yiranx/facedata/wildvideos_eg3d_simple/silverman1 --outdir out/optim_multi_dynamic/silverman1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose --num_epochs 500 --lr 5e-3 --weight_lpips 1.0 --weight_residual 0.7 --w_init_samples 500 --save_intermediates=True --random_init=True --use_mask=False --weight_delta_norm 1e-3 --use_pre_pose=True
"""

import os
import sys
sys.path.append(".")
sys.path.append("..")

import legacy
import dnnlib
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, angles2mat
from training.triplane import TriPlaneGenerator
from inversion_dataloader import MutilFrameDatasetDynamic as MutilFrameDataset

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

def nn_lpips_initialization(G, target_image, num_samples=2000, batch=1, cam_pose=None, lpips_backbone='alex', device='cuda'):
    lpips_fn = lpips.LPIPS(net=lpips_backbone, spatial=True).to(device).eval()
    target_image = target_image.unsqueeze(0).to(device)
    target_image = F.interpolate(target_image.clone(), size=(G.img_resolution, G.img_resolution), mode='bilinear')
    intrinsics = torch.tensor([[4.4652, 0.0000, 0.5000],
                                    [0.0000, 4.4652, 0.5000],
                                    [0.0000, 0.0000, 1.0000]], device=device)

    if cam_pose is None:
        ws_plus = []
        yaws = []
        pitchs = []
        camera_params_all = []
        lpips_scores = []
        print("Initializing w...")
        pitch_range = 0.25
        yaw_range = 0.35
        with torch.no_grad():
            for itr in tqdm(range(num_samples)):
                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

                sample_yaw = (yaw_range * torch.randn(batch, 1)).to(device)
                sample_pitch = (pitch_range * torch.randn(batch, 1)).to(device)
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

                # cam2world_pose = angles2mat(torch.pi/2 + sample_yaw, torch.pi/2 + sample_pitch, cam_pivot, batch_size=batch, radius=cam_radius, device=device)
                cam2world_pose = LookAtPoseSampler.sample(torch.pi/2 + sample_yaw, torch.pi/2 + sample_pitch, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(torch.pi/2, torch.pi/2, cam_pivot, batch_size=batch, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                w_plus = G.mapping(z, conditioning_params, truncation_psi=1.0, truncation_cutoff=14)
                synth_image = G.synthesis(w_plus, camera_params)['image']

                lpips_src = lpips_fn(target_image, synth_image).mean().detach().cpu()
                lpips_scores.append(lpips_src)


                ws_plus.append(w_plus.detach().cpu())
                yaws.append(sample_yaw.detach().cpu())
                pitchs.append(sample_pitch.detach().cpu())
                camera_params_all.append(camera_params.detach().cpu())
        
        del lpips_fn
        torch.cuda.empty_cache()
        best_idx = torch.argmin(torch.stack(lpips_scores))
        print('Nearest LPIPS distance: ', lpips_scores[best_idx])
        w_plus_best = ws_plus[best_idx]
        w_best = w_plus_best[0:1]
        yaw_best = yaws[best_idx]
        pitch_best = pitchs[best_idx]
        camera_params_best = camera_params_all[best_idx]
        return w_best, w_plus_best, yaw_best, pitch_best, camera_params_best
    else:
        cam_pose = cam_pose.unsqueeze(0).to(device)
        ws_plus = []
        camera_params_all = []
        lpips_scores = []
        print("Initializing w...")
        with torch.no_grad():
            for itr in tqdm(range(num_samples)):
                z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(torch.pi/2, torch.pi/2, cam_pivot, batch_size=batch, radius=cam_radius, device=device)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                w_plus = G.mapping(z, conditioning_params, truncation_psi=1.0, truncation_cutoff=14)
                synth_image = G.synthesis(w_plus, cam_pose)['image']

                lpips_src = lpips_fn(target_image, synth_image).mean().detach().cpu()
                lpips_scores.append(lpips_src)

                ws_plus.append(w_plus.detach().cpu())
                camera_params_all.append(cam_pose.detach().cpu())
        del lpips_fn
        torch.cuda.empty_cache()
        best_idx = torch.argmin(torch.stack(lpips_scores))
        print('Nearest LPIPS distance: ', lpips_scores[best_idx])
        w_plus_best = ws_plus[best_idx]
        w_best = w_plus_best[0:1]
        camera_params_best = camera_params_all[best_idx]
        return w_best, w_plus_best, camera_params_best
        
def project(
    G,
    target_dataset,
    latents_path,
    output_dir,
    device,
    num_frames                 = None,
    num_epochs                 = 500,
    w_init_samples             = 1000,
    batch_size                 = 1,
    initial_learning_rate      = 5e-3,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    weight_lpips               = 0.5,
    weight_residual            = 1.0,
    weight_delta_norm          = 2e-3,
    save_intermediates         = True,
    random_init                = True,
    use_raw_rgb_loss           = False,
    use_mask                   = True,
    use_inverse_mask           = False,
    use_pre_pose               = False,
):
    
     # loss functions
    if not use_mask:
        l2_fn = torch.nn.MSELoss(reduction='mean')
    else:
        l2_fn = torch.nn.MSELoss(reduction='sum')
    lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device).eval()   

    # Create dataset
    video_dataset = MutilFrameDataset(data_root=target_dataset, yaw_opt=None, pitch_opt=None, use_pre_pose=use_pre_pose)
    os.makedirs(output_dir, exist_ok=True)
    # get initial angles and latent codes, TODO: angles and synthesize
    if random_init:
        ws_opt = []
        ws_plus_opt = []
        yaw_opt = []
        pitch_opt = []
        cam_params_opt = []
        
        print("Initializing for {} frames...".format(len(video_dataset)))
        with torch.no_grad():
            for init_iter in range(len(video_dataset)):
                
                if not use_pre_pose:
                    w_curr, w_plus_curr, yaw_curr, pitch_curr, cam_params_curr = nn_lpips_initialization(G, video_dataset[init_iter][0], device=device, num_samples=w_init_samples)
                    yaw_opt.append(yaw_curr)
                    pitch_opt.append(pitch_curr)
                else:
                    w_curr, w_plus_curr, cam_params_curr = nn_lpips_initialization(G, video_dataset[init_iter][0], device=device, num_samples=w_init_samples, cam_pose=video_dataset[init_iter][2])
                ws_opt.append(w_curr)
                ws_plus_opt.append(w_plus_curr)
                cam_params_opt.append(cam_params_curr)
            
        ws_opt = torch.cat(ws_opt).mean(dim=0, keepdim=True)
        ws_plus_opt = torch.cat(ws_plus_opt).mean(dim=0, keepdim=True)
        cam_params_opt = torch.cat(cam_params_opt)
        if not use_pre_pose:
            yaw_opt = torch.cat(yaw_opt)
            pitch_opt = torch.cat(pitch_opt)
            init_vars = {'w': ws_opt.cpu(), 'w_plus': ws_plus_opt.cpu(), 'yaw': yaw_opt.cpu(), 'pitch': pitch_opt.cpu(), 'camera_params': cam_params_opt.cpu()}
        else:
            init_vars = {'w': ws_opt.cpu(), 'w_plus': ws_plus_opt.cpu(), 'camera_params': cam_params_opt.cpu()}
        
        torch.save(init_vars, os.path.join(output_dir, 'init_vars.pt'))
        ws_opt = ws_opt.clone().to(device).requires_grad_(True)
        ws_plus_opt = ws_plus_opt.clone().to(device).requires_grad_(True)
        if not use_pre_pose:
            yaw_opt = yaw_opt.clone().to(device).requires_grad_(True)
            pitch_opt = pitch_opt.clone().to(device).requires_grad_(True)
        cam_params_opt = cam_params_opt[:, :16].clone().to(device).requires_grad_(True)
        ws_plus_cano_opt = init_vars['w_plus'].clone().detach().cpu()[:, 0, :]
        ws_plus_cano_opt = ws_plus_cano_opt.clone().to(device).requires_grad_(True)  # kinda not efficient
        ws_plus_residual_opt = torch.zeros(cam_params_opt.shape[0], 512).to(device).requires_grad_(True)
    else:
        init_vars = torch.load(latents_path)
        ws_plus_opt = init_vars['w_plus'].clone().to(device).requires_grad_(True)
        if not use_pre_pose:
            yaw_opt = init_vars['yaw'].clone().to(device).requires_grad_(True)
            pitch_opt = init_vars['pitch'].clone().to(device).requires_grad_(True)
        cam_params_opt = init_vars['camera_params'][:, :16].clone().to(device).requires_grad_(True)
        ws_plus_cano_opt = init_vars['w_plus'].clone().detach().cpu()[:, 0:1, :]
        ws_plus_cano_opt = ws_plus_cano_opt.clone().to(device).requires_grad_(True)  # kinda not efficient
        ws_plus_residual_opt = torch.zeros(cam_params_opt.shape[0], 1, 512).to(device).requires_grad_(True)
    # ws_cano_opt = ws_plus_cano_opt
    # ws_residual_opt = ws_plus_residual_opt
    
    # set up data loader
    # Create dataset
    video_dataset = MutilFrameDataset(data_root=target_dataset, yaw_opt=None, pitch_opt=None, use_pre_pose=use_pre_pose)
    
    # set some optimized variables
    video_dataset.w_plus = ws_plus_residual_opt
    if not use_pre_pose:
        video_dataset.yaw_opt = yaw_opt
        video_dataset.pitch_opt = pitch_opt
        camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
    else:
        video_dataset.cam_params_json = cam_params_opt
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    # video_dataset.cam_params = cam_params_opt
    
    train_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # # prepare optimization variable
    # w_opt = w_render.clone().to(device).requires_grad_(True)
    
    # freeze G
    for param in G.parameters():
        param.requires_grad = False
    
    # optimizers
    print("Optimizing w...")
    optimizer_w = torch.optim.Adam([
                                {'params': [ws_plus_cano_opt], 'lr': float(initial_learning_rate)},
                                {'params': [ws_plus_residual_opt], 'lr':float(initial_learning_rate)},
                                # {'params': G2.generator.parameters(), 'lr': float(initial_learning_rate)},
                                # betas=(0.9, 0.999),
                                ])
    if not use_pre_pose:
        optimizer_angle = torch.optim.Adam([
                                    {'params': [yaw_opt], 'lr': float(initial_learning_rate)},
                                    {'params': [pitch_opt], 'lr': float(initial_learning_rate)},
                                    # {'params': [cam_params_opt], 'lr': float(initial_learning_rate)}
                                    ])
    # else:
    #     optimizer_angle = torch.optim.Adam([
    #                                 {'params': [cam_params_opt], 'lr': float(initial_learning_rate)}
    #                                 ])

    global_iters = 0

    if save_intermediates:
        os.makedirs(os.path.join(output_dir, 'intermediates'), exist_ok=True)

    loss_l2_epochs = []
    loss_lpips_epochs = []
    loss_delta_norm_epochs = []
    loss_total_epochs = []
    step_w = True
    step_angle = False if use_pre_pose else True
    for epoch in range(num_epochs):
        # Learning rate schedule.
        t = epoch / num_epochs
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        
        if step_w:
            for param_group in optimizer_w.param_groups:
                param_group['lr'] = lr
        if step_angle:
            for param_group in optimizer_angle.param_groups:
                param_group['lr'] = lr

        loss_l2_iters = []
        loss_lpips_iters = []
        loss_delta_norm_iters = []
        loss_total_iters = []
        
        for itr, data in enumerate(train_dataloader):
            # target_images, masks_tensor, camera_params_opt_curr, w_plus_residual_opt_curr, _ = data
            # target_images = target_images.to(device)
            # target_images = F.interpolate(target_images, size=(G.img_resolution, G.img_resolution), mode='bilinear')
            # # cam_intrinsics = cam_intrinsics.to(device)
            # yaw_opt_curr = yaw_opt_curr.to(device)
            # pitch_opt_curr = pitch_opt_curr.to(device)
            # # camera_params_opt_curr = camera_params_opt_curr.to(device)
            if not use_pre_pose:
                target_images, masks_tensor, _, yaw_opt_curr, pitch_opt_curr, w_plus_residual_opt_curr, _ = data
                target_images = target_images.to(device)
                target_images = F.interpolate(target_images, size=(G.img_resolution, G.img_resolution), mode='bilinear')
                # yaw_opt_curr = yaw_opt_curr.to(device)
                # pitch_opt_curr = pitch_opt_curr.to(device)
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_opt_curr,
                                                        3.14/2 + pitch_opt_curr,
                                                        camera_lookat_point, radius=2.7, device=device)
            else:
                target_images, masks_tensor, cam2world_pose, w_plus_residual_opt_curr, _ = data
                target_images = target_images.to(device)
                target_images = F.interpolate(target_images, size=(G.img_resolution, G.img_resolution), mode='bilinear')
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            if use_mask:
                masks_tensor = masks_tensor.to(device)
                if use_inverse_mask:
                    masks_tensor = 1.0 - masks_tensor
                masks_tensor = F.interpolate(masks_tensor, size=(G.img_resolution, G.img_resolution), mode='nearest')

            if use_raw_rgb_loss:
                target_images_downsampled = F.interpolate(target_images.clone(), size=(64, 64), mode='bilinear')

            # compute extrinsics
            # cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            # cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            # cam2world_pose = LookAtPoseSampler.sample(torch.pi/2 + yaw_opt_curr, torch.pi/2 + pitch_opt_curr, cam_pivot, radius=cam_radius, device=device)
            # conditioning_cam2world_pose = LookAtPoseSampler.sample(torch.pi/2, torch.pi/2, cam_pivot, radius=cam_radius, device=device)
            
            # camera_params = torch.cat([cam2world_pose.reshape(-1, 16), cam_intrinsics.reshape(-1, 9)], 1)
            # conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), cam_intrinsics.reshape(-1, 9)], 1)
            
            # forward-pass: generate images
            w_plus_opt_curr = ws_plus_cano_opt + weight_residual * w_plus_residual_opt_curr
            w_plus_opt_curr = w_plus_opt_curr.repeat(1, 14, 1)

            synth_images = G.synthesis(w_plus_opt_curr, camera_params)['image']
            
            loss_dict = {}
            loss = 0.0
            if not use_mask:
                # LPIPS distance
                lpips_loss = lpips_fn(target_images, synth_images).mean()
                
                # L2 loss
                l2_loss = l2_fn(target_images, synth_images).mean()
            else:
                # LPIPS distance
                lpips_loss = (lpips_fn(target_images, synth_images) * masks_tensor).sum()/masks_tensor.sum()
                
                # L2 loss
                l2_loss = l2_fn(target_images * masks_tensor, synth_images * masks_tensor)/masks_tensor.sum()
            
            # delta loss
            # total_delta_loss = 0.0
            # if weight_delta_norm > 0.0:
            #     first_w = w_plus_opt_curr[:, 0, :]
            #     for i in range(1, 14 - 1):
            #         # curr_dim = deltas_latent_dims[i]
            #         curr_dim = i
            #         delta = w_plus_opt_curr[:, curr_dim, :] - first_w
            #         delta_loss = torch.norm(delta, 2, dim=1).mean()
            #         # loss_dict[f"delta{i}_loss"] = float(delta_loss)
            #         total_delta_loss += delta_loss
            #     loss_dict['total_delta_loss'] = float(total_delta_loss)
            #     loss += weight_delta_norm * total_delta_loss

            # record loss values
            loss_dict['LPIPS'] = lpips_loss.item()
            loss_dict['L2'] = l2_loss.item()

            loss += weight_lpips * lpips_loss + l2_loss

            if use_raw_rgb_loss:
                if not use_mask:
                    raw_lpips = lpips_fn(target_images_downsampled, thumb_rgb).mean()
                    raw_l2 = l2_fn(target_images_downsampled, thumb_rgb)
                else:
                    mask_downsampled = F.interpolate(masks_tensor.clone(), size=(64, 64), mode='bilinear')
                    raw_lpips = (lpips_fn(target_images_downsampled , thumb_rgb) * mask_downsampled).sum()/mask_downsampled.sum()
                    raw_l2 = l2_fn(target_images_downsampled * mask_downsampled, thumb_rgb * mask_downsampled)/mask_downsampled.sum()

                loss += weight_lpips * raw_lpips + raw_l2
                loss_dict['RAW_LPIPS'] = raw_lpips.item()
                loss_dict['RAW_L2'] = raw_l2.item()

            
            # Step
            # optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # optimizer.step()
            if step_w and not step_angle:
                optimizer_w.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_w.step()
            elif step_angle and not step_w:
                optimizer_angle.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_angle.step()
            elif step_w and step_angle:
                optimizer_w.zero_grad(set_to_none=True)
                optimizer_angle.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_w.step()
                optimizer_angle.step()

            print_txt = f'Epoch {epoch+1:>4d}/{num_epochs}, Itr {global_iters+1}: '
            for loss_key in loss_dict:
                print_txt += loss_key + f' {loss_dict[loss_key]:<4.2f} ' 
            print_txt += f' loss {float(loss):<5.2f}'
            print(print_txt)
            
            # save loss values every iteration
            loss_l2_iters.append(l2_loss.item())
            loss_lpips_iters.append(lpips_loss.item())
            # if weight_delta_norm > 0.0 and torch.is_tensor(total_delta_loss):
            #     loss_delta_norm_iters.append(total_delta_loss.item())
            # else:
            #     loss_delta_norm_iters.append(0.0)
            loss_total_iters.append(loss.item())

            # save intermediate result
            if save_intermediates and ((global_iters + 1) % 500 == 0 or global_iters == 0):  # debug
            # if save_intermediates and data[-1][0] == '/home/yiranx/data/capvideo_simple/11986_16f/frames/frame0203.png':
                with torch.no_grad():
                    # synth_image = G.synthesis(w_plus_opt_curr, camera_params_opt_curr)['image']
                    # # for canonical frame
                    # cano_image = G.synthesis(ws_plus_cano_opt, camera_params_opt_curr)['image']

                    synth_image = G.synthesis(w_plus_opt_curr, camera_params)['image']
                    # for canonical frame
                    cano_image = G.synthesis(ws_plus_cano_opt.repeat(1, 14, 1), camera_params)['image']

                    synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    cano_image = F.interpolate(cano_image, size=(256, 256), mode='area')
                    cano_image = (cano_image + 1) * (255/2)
                    cano_image = cano_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    target_image = F.interpolate(target_images[0:1], size=(256, 256), mode='area')
                    target_image = (target_image + 1) * (255/2)
                    target_image = target_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    output_image = np.concatenate((cano_image, synth_image, target_image), axis=1)
                    imageio.imwrite(os.path.join(output_dir, 'intermediates', 'step_{:05}.jpg'.format(global_iters+1) ), output_image)

            global_iters += 1
        # if epoch == 500:
        #     print("Optimizing both w and angles...")
        #     step_w = True
        #     step_angle = True
        # if epoch in [100, 200, 300, 400, 500, 600, 700]:
        #     print("Optimizing both w and angles...")
        #     step_w = True
        #     step_angle = True
        
        # if epoch in [150, 250, 350, 450, 550, 650]:
        #     print("Optimizing both w and angles...")
        #     step_w = False
        #     step_angle = True

        loss_l2_epochs.append(np.mean(loss_l2_iters))
        loss_lpips_epochs.append(np.mean(loss_lpips_iters))
        # loss_delta_norm_epochs.append(np.mean(loss_delta_norm_iters) * 0.01)
        loss_total_epochs.append(np.mean(loss_total_iters))

        plt.figure()
        plt.plot(loss_l2_epochs, color='green', marker='o',label='L2')
        plt.plot(loss_lpips_epochs, color='blue', marker='o',label='LPIPS')
        # plt.plot(loss_delta_norm_epochs, color='purple', marker='o', label='Delta_norm/100')
        plt.plot(loss_total_epochs, color='red', marker='o',label='Total')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(output_dir, 'training_loss.jpg'))
        plt.close('all')
    if not use_pre_pose:
        return ws_plus_cano_opt, ws_plus_residual_opt, yaw_opt, pitch_opt, cam_params_opt
    else:
        return ws_plus_cano_opt, ws_plus_residual_opt, None, None, cam_params_opt
    

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target_path', help='Target frame path', type=str, required=True, metavar='DIR')
@click.option('--latents_path', help='latent code path', type=str, )
@click.option('--outdir', help='Where to save the output results', type=str, required=True, metavar='DIR')
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--num_epochs', help='Number of epochs', type=int, default=500, show_default=True)
@click.option('--lr', help='learning rate', type=float, metavar='float', default=1e-2, show_default=True)
@click.option('--weight_lpips', help='weight of lpips loss', type=float, metavar='float', default=0.5, show_default=True)
@click.option('--weight_residual', help='weight of residual', type=float, metavar='float', default=1.0, show_default=True)
@click.option('--weight_delta_norm', help='weight of delta_norm', type=float, metavar='float', default=2e-3, show_default=True)
@click.option('--w_init_samples', type=int, help='Samples for initialization', default=2000)
@click.option('--batch_size', type=int, help='batch size', default=1)
@click.option('--save_intermediates', type=bool, help='if save intermediate results', default=True, show_default=True)
@click.option('--random_init', type=bool, help='if randomly initialize variables', default=True, show_default=True)
@click.option('--use_raw_rgb_loss', type=bool, help='if use raw rgb', default=False, show_default=True)
@click.option('--use_mask', type=bool, help='if use mask', default=False, show_default=True)
@click.option('--use_inverse_mask', type=bool, help='if use inverse mask', default=False, show_default=True)
@click.option('--use_pre_pose', type=bool, help='if use pre-estimated pose as intialization', default=False, show_default=True)


def run_projection(
    network_pkl,
    target_path,
    latents_path,
    outdir,
    fov_deg=18.837,
    num_epochs=500,
    lr=1e-2,
    weight_lpips=0.5,
    weight_residual=1.0,
    weight_delta_norm=2e-3,
    w_init_samples=2000,
    batch_size=1,
    save_intermediates=True,
    random_init=True,
    use_raw_rgb_loss=False,
    use_mask=True,
    use_inverse_mask=False,
    use_pre_pose=False,
):
    
    device = torch.device('cuda')
    # load a generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # run projection
    ws_plus_cano_opt, ws_plus_residual_opt, yaw_opt, pitch_opt, camera_params_opt = project(
                                                                        G,
                                                                        target_path,
                                                                        latents_path,
                                                                        output_dir=outdir,
                                                                        device=device,
                                                                        num_frames                 = None,
                                                                        num_epochs                 = num_epochs,
                                                                        w_init_samples             = w_init_samples,
                                                                        batch_size                 = batch_size,
                                                                        initial_learning_rate      = lr,
                                                                        weight_lpips               = weight_lpips,
                                                                        weight_residual            = weight_residual,
                                                                        weight_delta_norm          = weight_delta_norm,
                                                                        save_intermediates         = save_intermediates,
                                                                        random_init                = random_init,
                                                                        use_raw_rgb_loss           = use_raw_rgb_loss,
                                                                        use_mask                   = use_mask,
                                                                        use_inverse_mask           = use_inverse_mask,
                                                                        use_pre_pose               = use_pre_pose,
                                                                    )

    os.makedirs(os.path.join(outdir, 'frames', 'projected'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'targets'), exist_ok=True)


    # Save final projected frame and W vector,
    video_dataset = MutilFrameDataset(data_root=target_path, yaw_opt=None, pitch_opt=None, use_pre_pose=use_pre_pose)
    
    video_dataset.w_plus = ws_plus_residual_opt
    if not use_pre_pose:
        video_dataset.yaw_opt = yaw_opt
        video_dataset.pitch_opt = pitch_opt
    else:
        video_dataset.cam_params_json = camera_params_opt
    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    # video_dataset.cam_params = cam_params_opt
    eval_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    
    print("Saving results...")
    f = open(os.path.join(outdir, 'target_frames.txt'), 'w')

    with torch.no_grad():
        for itr, data in enumerate(eval_dataloader):
            if not use_pre_pose:
                target_image, masks_tensor, _, yaw_opt_curr, pitch_opt_curr, w_plus_residual_opt_curr, img_path = data
                target_image = target_image.to(device)
                target_image = F.interpolate(target_image, size=(G.img_resolution, G.img_resolution), mode='bilinear')
                # yaw_opt_curr = yaw_opt_curr.to(device)
                # pitch_opt_curr = pitch_opt_curr.to(device)
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_opt_curr,
                                                        3.14/2 + pitch_opt_curr,
                                                        camera_lookat_point, radius=2.7, device=device)
            else:
                target_image, masks_tensor, cam2world_pose, w_plus_residual_opt_curr, img_path = data
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            w_plus_opt_curr = ws_plus_cano_opt + weight_residual * w_plus_residual_opt_curr
            w_plus_opt_curr = w_plus_opt_curr.repeat(1, 14, 1)
            synth_image = G.synthesis(w_plus_opt_curr, camera_params)['image']
        

            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            target_image = (target_image + 1) * (255/2)
            target_image = target_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                

            PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'frames', 'projected', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(target_image, 'RGB').save(os.path.join(outdir, 'frames', 'targets', 'target_{0:05d}.png'.format(itr)))
            # PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'frames', 'projected', 'proj_'+os.path.basename(img_path[0])))
            # PIL.Image.fromarray(target_image, 'RGB').save(os.path.join(outdir, 'frames', 'targets', 'target_'+os.path.basename(img_path[0])))
            f.write(os.path.basename(img_path[0]) + '\n')

            if itr == 0: # visualize canonical frame
                # synth_image = G.synthesis(ws_plus_cano_opt, camera_params_opt_curr)['image']
                synth_image = G.synthesis(ws_plus_cano_opt.repeat(1, 14, 1), camera_params)['image']

                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'frames', 'cano_{0:05d}.png'.format(itr)))

    f.close()

    torch.save({'w_plus': (ws_plus_cano_opt+weight_residual*ws_plus_residual_opt).detach().cpu(),
                'w_plus_cano': ws_plus_cano_opt.detach().cpu(), 'w_plus_residual': ws_plus_residual_opt.detach().cpu(),
                }, 
                f'{outdir}/latents.pt')
    
    cam_intrinsics = torch.tensor([[4.4652, 0.0000, 0.5000],
                            [0.0000, 4.4652, 0.5000],
                            [0.0000, 0.0000, 1.0000]])
    if not use_pre_pose:
        torch.save({'yaw': yaw_opt.cpu(), 'pitch': pitch_opt.cpu(), 'cam_intrinsics': cam_intrinsics, 'cam_params': camera_params_opt.detach().cpu()}, f'{outdir}/cam_params.pt')
    else:
        torch.save({'cam_intrinsics': cam_intrinsics, 'cam_params': camera_params_opt.detach().cpu()}, f'{outdir}/cam_params.pt')
    # torch.save({camera_params_opt.cpu()}, f'{outdir}/cam_params.pt')
#----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection()