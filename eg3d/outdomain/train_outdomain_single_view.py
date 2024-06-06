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
from outdomain.loss_functions import compute_blendw_loss, compute_blendw_area_loss, compute_sparse_loss
from outdomain.depth_alignment import calibrate_disparity, render_depth
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
@click.option('--ckpt_path', help='checkpoint path', required=False, default=None)
@click.option('--target_path', help='Target frame path', type=str, required=True, metavar='DIR')
@click.option('--latents_path', help='latent code path', type=str, )
@click.option('--outdir', help='Where to save the output results', type=str, required=True, metavar='DIR')
@click.option('--num_epochs_raw', help='Number of epochs', type=int, default=500000, show_default=True)
@click.option('--max_num_iters_raw', help='maxium iterations', type=int, default=20000, show_default=True)
@click.option('--num_epochs_sr', help='Number of epochs', type=int, default=500, show_default=True)

@click.option('--lr', help='learning rate', type=float, metavar='float', default=1e-2, show_default=True)
@click.option('--lr_sr', help='learning rate', type=float, metavar='float', default=1e-6, show_default=True)
# @click.option('--weight_lpips', help='weight of lpips loss', type=float, metavar='float', default=0.5, show_default=True)
# @click.option('--weight_residual', help='weight of residual', type=float, metavar='float', default=1.0, show_default=True)
# @click.option('--weight_delta_norm', help='weight of delta_norm', type=float, metavar='float', default=2e-3, show_default=True)
# @click.option('--w_init_samples', type=int, help='Samples for initialization', default=2000)
@click.option('--batch_size', type=int, help='batch size', default=1)
@click.option('--save_intermediates', type=bool, help='if save intermediate results', default=True, show_default=True)
@click.option('--vis_step', type=int, help='batch size', default=50)
@click.option('--random_init', type=bool, help='if randomly initialize variables', default=True, show_default=True)
@click.option('--use_raw_rgb_loss', type=bool, help='if use raw rgb', default=False, show_default=True)
@click.option('--use_mask', type=bool, help='if use mask', default=False, show_default=True)
@click.option('--return_raw_blendw', type=bool, help='if return blendw_coarse and fine', default=False, show_default=True)
@click.option('--lamb_l2_raw_full', type=float, help='weight of l2 loss for raw rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_l2_raw_o', type=float, help='weight of l2 loss for out-of-distribution part', default=1.0, show_default=True)
@click.option('--lamb_lpips_raw_full', type=float, help='weight of lpips loss for raw rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_lpips_raw_o', type=float, help='weight of lpips loss for out-of-distribution part', default=1.0, show_default=True)
@click.option('--lamb_l2_sr_full', type=float, help='weight of l2 loss for SR rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_lpips_sr_full', type=float, help='weight of l2 loss for SR rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_l2_sr_o', type=float, help='weight of l2 loss for SR rgb out-of-distribution part', default=1.0, show_default=True)
@click.option('--lamb_blendw_loss', type=float, help='weight of blending weights out-of-distribution part', default=1.0, show_default=True)
@click.option('--blendw_skew', type=float, help='skewness for computing blending weight loss', default=2.0, show_default=True)
@click.option('--lamb_blendw_area_loss', type=float, help='weight of area blending weights out-of-distribution part', default=1.0, show_default=True)
@click.option('--lamb_blendw_sparse_loss', type=float, help='weight of sparse blending weights out-of-distribution part', default=0.0, show_default=True)
@click.option('--lamb_depth_reg', type=float, help='weight of depth regularization', default=0.0, show_default=True)
@click.option('--lamb_dist_loss', type=float, help='weight of distortion loss', default=0.0, show_default=True)
@click.option('--train_raw', type=bool, help='if train raw resolution module', default=True, show_default=True)
@click.option('--train_sr', type=bool, help='if train sr module', default=False, show_default=True)
@click.option('--comp_dist_loss', type=bool, help='if compute distortion loss', default=False, show_default=True)
@click.option('--min_step_use_lpips_mask', type=int, help='minimum step to use lpips mask', default=1000000, show_default=True)
@click.option('--save_ckpt_step', type=int, help='step to save ckpt', default=2000, show_default=True)

def outdomain_inv(
        network_pkl,
        ckpt_path,
        target_path,
        latents_path,
        outdir,
        num_epochs_raw=500,
        max_num_iters_raw=20000,
        num_epochs_sr=200,
        lr=1e-3,
        lr_sr=1e-6,
        # weight_lpips=0.5,
        # weight_residual=1.0,
        # weight_delta_norm=2e-3,
        # w_init_samples=2000,
        batch_size=1,
        save_intermediates=True,
        vis_step=50,
        random_init=True,
        use_raw_rgb_loss=False,
        use_mask=True,
        return_raw_blendw=True,
        lamb_l2_raw_full=1.0,
        lamb_l2_raw_o=1.0,
        lamb_lpips_raw_full=1.0,
        lamb_lpips_raw_o=1.0,
        lamb_l2_sr_full=1.0,
        lamb_lpips_sr_full=1.0,
        lamb_l2_sr_o=1.0,
        lamb_blendw_loss=1.0,
        blendw_skew=2.0,
        lamb_blendw_area_loss=1.0,
        lamb_blendw_sparse_loss=1.0,
        lamb_depth_reg=0.0,
        lamb_dist_loss=0.0,
        comp_dist_loss=False,
        train_raw=True,
        train_sr=True,
        min_step_use_lpips_mask=1000000,
        save_ckpt_step=2000,
    ):
    ori_use_mask = use_mask
    # load generator
    device = torch.device('cuda')
    # load a generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # load initial latent codes (from other inversion method)
    latents = torch.load(latents_path)
    # w_plus = latents['w_plus']
    w_plus = latents['w_plus'].clone().to(device).requires_grad_(True)
    
    # randomly initialize phi codes
    phi = torch.randn(w_plus.shape[0], 32).clone().to(device).requires_grad_(True)
    
    # build composed triplanes for the out-of-distribution item
    compose_triplanes = ComposeTriplane(G, comp_dist_loss=comp_dist_loss).to(device)
    if ckpt_path is not None:
        print("Loading from ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        compose_triplanes.load_state_dict(ckpt['compose_triplanes'])
        phi = ckpt['phi']

    # build dataset and dataloader
    video_dataset = MutilFrameDataset(data_root=target_path, yaw_opt=None, pitch_opt=None, use_pre_pose=True)
    video_dataset.w_plus = w_plus
    video_dataset.phi = phi
    # Will return img_tensor, mask_tensor, cam_param, w_plus_opt_curr, img_path
    train_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if train_raw:
        for param in compose_triplanes.parameters():
            param.requires_grad = True

        # freeze G
        for param in G.parameters():
            param.requires_grad = False
        # for param in G.superresolution.parameters():
        #     param.requires_grad = True
        # G.eval()

        # build optimizers
        optimizer = torch.optim.Adam([
            {'params': compose_triplanes.new_triplanes.parameters(), 'lr': float(lr)},
            {'params': phi, 'lr': float(lr)},
            # {'params': w_plus, 'lr': float(lr)/5000.0},
            # {'params': compose_triplanes.G.superresolution.parameters(), 'lr': float(lr/100)}
        ])
        
        compose_triplanes.train()


        # build criterions
        if not use_mask:
            l2_fn = torch.nn.MSELoss(reduction='mean')
        else:
            l2_fn = torch.nn.MSELoss(reduction='sum')
        lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device).eval()   

        # start training
        global_iters = 0
        # lr_rampdown_length = 0.25
        # lr_rampup_length = 0.05
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        if save_intermediates:
            os.makedirs(os.path.join(outdir, 'intermediates'), exist_ok=True)

        loss_l2_raw_full_epochs = []
        loss_l2_raw_o_epochs = []
        loss_lpips_raw_full_epochs = []
        loss_lpips_raw_o_epochs = []
        loss_l2_sr_full_epochs = []
        loss_l2_sr_o_epochs = []
        loss_blendw_epochs = []
        loss_blendw_area_epochs = []
        loss_blendw_sparse_epochs = []
        loss_depth_epochs = []
        loss_dist_epochs = []
        loss_total_epochs = []
        num_epochs = num_epochs_raw
        training_flag = True
        for epoch in range(num_epochs):
            if not training_flag:
                break
            # Learning rate schedule.
            # t = epoch / num_epochs
            # lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            # lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            # lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            # lr = lr * lr_ramp
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            loss_l2_raw_full_iters = []
            loss_l2_raw_o_iters = []
            loss_lpips_raw_full_iters = []
            loss_lpips_raw_o_iters = []
            loss_l2_sr_full_iters = []
            loss_l2_sr_o_iters = []
            loss_blendw_iters = []
            loss_blendw_area_iters = []
            loss_blendw_sparse_iters = []
            loss_depth_iters = []
            loss_dist_iters = []
            loss_total_iters = []

            for itr, data in enumerate(train_dataloader):
                if video_dataset.depth_maps is not None:
                    target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, depth_map_est, _ = data
                else:
                    target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, _ = data
                target_images = target_images.to(device)
                target_images = F.interpolate(target_images, size=(G.img_resolution, G.img_resolution), mode='bilinear')
                camera_params = camera_params.to(device)
                w_plus_curr = w_plus_curr.to(device)
                phi_curr = phi_curr.to(device)

                if global_iters == min_step_use_lpips_mask:
                    use_mask = True
                    with torch.no_grad():
                        masks_tensor = lpips_fn(synth_images_full, target_images).detach()
                        masks_tensor = 1 - masks_tensor
                elif global_iters > min_step_use_lpips_mask:
                    use_mask = ori_use_mask

                if use_mask:
                    masks_tensor = masks_tensor.to(device)
                    masks_tensor = F.interpolate(masks_tensor, size=(G.img_resolution, G.img_resolution), mode='nearest')

                if use_raw_rgb_loss:
                    target_images_downsampled = F.interpolate(target_images.clone(), size=(128, 128), mode='bilinear')
                    if use_mask:
                        mask_downsampled = F.interpolate(masks_tensor, size=(128, 128), mode='nearest')

                # forward pass
                out = compose_triplanes(w_plus_curr, camera_params, phi_curr, masks_tensor, return_raw_blendw=return_raw_blendw)
                
                synth_images_full = out['image_full']
                thumb_rgb_full = out['image_raw_full']
                synth_images_o = out['image_o']
                thumb_rgb_o = out['image_raw_o'] 
                blendw_coarse = out['blendw_coarse']
                blendw_fine = out['blendw_fine']
                depth_map = out['image_depth_full']
                
                loss_dict = {}
                loss = 0.0

                if lamb_l2_raw_full > 0.0:
                    l2_raw_full = l2_fn(thumb_rgb_full, target_images_downsampled)/(thumb_rgb_full.shape[0] * thumb_rgb_full.shape[1] * thumb_rgb_full.shape[2] * thumb_rgb_full.shape[3])
                    loss += lamb_l2_raw_full * l2_raw_full
                    loss_dict["RAW_L2_FULL"] = l2_raw_full.item()
                    loss_l2_raw_full_iters.append(loss_dict["RAW_L2_FULL"])
                else:
                    loss_l2_raw_full_iters.append(0.0)
                    
                if lamb_l2_raw_o > 0.0 and use_mask:
                    l2_raw_o = l2_fn(thumb_rgb_o * (1 - mask_downsampled), target_images_downsampled * (1 - mask_downsampled))/(1 - mask_downsampled).sum()
                    # l2_raw_o = l2_fn(thumb_rgb_o, target_images_downsampled)/(thumb_rgb_o.shape[0] * thumb_rgb_o.shape[1] * thumb_rgb_o.shape[2] * thumb_rgb_o.shape[3])
                    loss += lamb_l2_raw_o * l2_raw_o
                    loss_dict["RAW_L2_O"] = l2_raw_o.item()
                    loss_l2_raw_o_iters.append(loss_dict["RAW_L2_O"])
                else:
                    loss_l2_raw_o_iters.append(0.0)

                if lamb_lpips_raw_full > 0.0:
                    lpips_raw_full = (lpips_fn(thumb_rgb_full, target_images_downsampled)).mean()
                    loss += lpips_raw_full * lamb_lpips_raw_full
                    loss_dict["RAW_LPIPS_FULL"] = lpips_raw_full.item()
                    loss_lpips_raw_full_iters.append(loss_dict["RAW_LPIPS_FULL"])
                else:
                    loss_lpips_raw_full_iters.append(0.0)

                if lamb_lpips_raw_o > 0.0 and use_mask:
                    lpips_raw_o = (lpips_fn(thumb_rgb_o, target_images_downsampled) * (1 - mask_downsampled)).sum()/(1 - mask_downsampled).sum()
                    loss += lpips_raw_o * lpips_raw_o
                    loss_dict["RAW_LPIPS_O"] = lpips_raw_o.item()
                    loss_lpips_raw_o_iters.append(loss_dict["RAW_LPIPS_O"])
                else:
                    loss_lpips_raw_o_iters.append(0.0)
    
                if lamb_l2_sr_full > 0.0:
                    l2_sr_full = l2_fn(synth_images_full, target_images)/(target_images.shape[0] * target_images.shape[1] * target_images.shape[2] * target_images.shape[3])
                    loss += lamb_l2_sr_full * l2_sr_full
                    loss_dict["SR_FULL"] = l2_sr_full.item()
                    loss_l2_sr_full_iters.append(loss_dict["SR_FULL"])
                else:
                    loss_l2_sr_full_iters.append(0.0)

                if lamb_l2_sr_o > 0.0 and use_mask: 
                    l2_sr_o = l2_fn(target_images * (1 - masks_tensor), synth_images_o * (1 - masks_tensor))/(1 - masks_tensor).sum()
                    loss += lamb_l2_sr_o * l2_sr_o
                    loss_dict["SR_O"] = l2_sr_o.item()
                    loss_l2_sr_o_iters.append(loss_dict["SR_O"])
                else:
                    loss_l2_sr_o_iters.append(0.0)

                if lamb_blendw_loss > 0.0:
                    blendw_loss = compute_blendw_loss(blendw_coarse, blendw_fine, skewness=blendw_skew).mean()
                    loss += lamb_blendw_loss * blendw_loss
                    loss_dict["BLENDW_LOSS"] = blendw_loss.item()
                    loss_blendw_iters.append(loss_dict["BLENDW_LOSS"])
                else:
                    loss_blendw_iters.append(0.0)

                if lamb_blendw_area_loss > 0.0:
                    blendw_area_loss = compute_blendw_area_loss(blendw_coarse, blendw_fine)
                    loss += lamb_blendw_area_loss * blendw_area_loss
                    loss_dict["BLENDW_AREA_LOSS"] = blendw_area_loss.item()
                    loss_blendw_area_iters.append(loss_dict["BLENDW_AREA_LOSS"])
                else:
                    loss_blendw_area_iters.append(0.0)

                if lamb_blendw_sparse_loss > 0.0 and use_mask:
                    blendw_sparse_loss = compute_sparse_loss(blendw_coarse, blendw_fine, mask_downsampled[:, 0:1].permute(0,2,3,1).reshape(mask_downsampled.shape[0], -1, 1))
                    loss += lamb_blendw_sparse_loss * blendw_sparse_loss
                    loss_dict["BLEND_SPARSE_LOSS"] = blendw_sparse_loss.item()
                    loss_blendw_sparse_iters.append(loss_dict["BLEND_SPARSE_LOSS"])
                else:
                    loss_blendw_sparse_iters.append(0.0)
                
                if lamb_depth_reg > 0.0:
                    # align depth
                    depth_map_est = F.interpolate(depth_map_est.unsqueeze(1), size=(depth_map.shape[2], depth_map.shape[3]), mode='bilinear', align_corners=True)
                    depth_map_aligned = calibrate_disparity(depth_map_est.squeeze(), depth_map.squeeze()).to(device)
                    # Charbonnier penalty for output depth `depth_map_aligned` and off-the-shelf depth `depth_map`
                    depth_reg = torch.mean(torch.sqrt(torch.pow(depth_map_aligned.squeeze() - depth_map.squeeze(), 2) + 1e-6))
                    # depth_reg = torch.sqrt(torch.pow(depth_map_aligned.squeeze() - depth_map.squeeze(), 2) + 1e-6)
                    # masks_tensor = F.interpolate(masks_tensor, size=(depth_map.shape[2], depth_map.shape[3]), mode='bilinear', align_corners=True).to(device)
                    # depth_reg = torch.sum(depth_reg * (1 - masks_tensor.squeeze()))/((1 - masks_tensor.squeeze()).sum() + 1e-6)
                    loss += lamb_depth_reg * depth_reg
                    loss_dict["DEPTH_REG"] = depth_reg.item()
                    loss_depth_iters.append(loss_dict["DEPTH_REG"])
                else:
                    loss_depth_iters.append(0.0)
                
                if comp_dist_loss and lamb_dist_loss > 0.0:
                    dist_loss = out['dist_loss']
                    loss += lamb_dist_loss * dist_loss
                    loss_dict["DIST_LOSS"] = dist_loss.item()
                    loss_dist_iters.append(loss_dict["DIST_LOSS"])
                else:
                    loss_dist_iters.append(0.0)


                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # print_txt = f'Epoch {epoch+1:>4d}/{num_epochs}, Itr {global_iters+1}: '
                print_txt = f'Epoch {epoch+1:>4d}, Itr {global_iters+1}/{max_num_iters_raw}: '
                for loss_key in loss_dict:
                    print_txt += loss_key + f' {loss_dict[loss_key]:<4.2f} ' 
                print_txt += f' loss {float(loss):<5.2f}'
                print(print_txt)
                            
                loss_total_iters.append(loss.item())

                # save intermediate result
                if save_intermediates and ((global_iters + 1) % vis_step == 0 or global_iters == 0):  # debug
                # if save_intermediates and data[-1][0] == '/home/yiranx/data/capvideo_simple/11986_16f/frames/frame0203.png':
                    with torch.no_grad():
                        out = compose_triplanes(w_plus_curr, camera_params, phi_curr, masks_tensor)
                        synth_image_full = out['image_raw_full']
                        synth_image_o = out['image_raw_o']
                        synth_image_full_sr = out['image_full']
                        synth_image_o_sr = out['image_o']


                        synth_image_full = F.interpolate(synth_image_full, size=(128, 128), mode='area')
                        synth_image_full = (synth_image_full + 1) * (255/2)
                        synth_image_full = synth_image_full.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                        
                        synth_image_o = F.interpolate(synth_image_o, size=(128, 128), mode='area')
                        synth_image_o = (synth_image_o + 1) * (255/2)
                        synth_image_o = synth_image_o.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                
                        synth_image_o_sr = F.interpolate(synth_image_o_sr, size=(128, 128), mode='area')
                        synth_image_o_sr = (synth_image_o_sr + 1) * (255/2)
                        synth_image_o_sr = synth_image_o_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                        synth_image_full_sr = F.interpolate(synth_image_full_sr, size=(128, 128), mode='area')
                        synth_image_full_sr = (synth_image_full_sr + 1) * (255/2)
                        synth_image_full_sr = synth_image_full_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()


                        synth_image_f = compose_triplanes.G.synthesis(w_plus_curr, camera_params)['image_raw']
                        synth_image_f = (synth_image_f + 1) * (255/2)
                        synth_image_f = synth_image_f.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                        
                        target_image = F.interpolate(target_images[0:1], size=(128, 128), mode='area')
                        target_image = (target_image + 1) * (255/2)
                        target_image = target_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                        depth_image = render_depth(depth_map.squeeze().cpu())

                        # ood rgb, full composite rgb, target
                        output_image = np.concatenate((synth_image_f, synth_image_o, synth_image_full, synth_image_o_sr, synth_image_full_sr, depth_image, target_image), axis=1)
                        imageio.imwrite(os.path.join(outdir, 'intermediates', 'step_{:05}.jpg'.format(global_iters+1) ), output_image)

                # save ckpt
                if (global_iters + 1) % save_ckpt_step == 0:
                    torch.save({'w_plus': w_plus.cpu(),
                                'phi': phi.cpu(),
                                'compose_triplanes': compose_triplanes.state_dict(),
                                }, f'{outdir}/triplanes{global_iters}.pt')

                global_iters += 1
                if global_iters >= max_num_iters_raw:
                    training_flag = False
                    break
            
            loss_l2_raw_full_epochs.append(np.mean(loss_l2_raw_full_iters))
            loss_l2_raw_o_epochs.append(np.mean(loss_l2_raw_o_iters))
            loss_lpips_raw_full_epochs.append(np.mean(loss_lpips_raw_full_iters))
            loss_lpips_raw_o_epochs.append(np.mean(loss_lpips_raw_o_iters))
            loss_l2_sr_full_epochs.append(np.mean(loss_l2_sr_full_iters))
            loss_l2_sr_o_epochs.append(np.mean(loss_l2_sr_o_iters))
            loss_blendw_epochs.append(np.mean(loss_blendw_iters))
            loss_blendw_area_epochs.append(np.mean(loss_blendw_area_iters))
            loss_blendw_sparse_epochs.append(np.mean(loss_blendw_sparse_iters))
            loss_depth_epochs.append(np.mean(loss_depth_iters))
            loss_total_epochs.append(np.mean(loss_total_iters))

            plt.figure()
            plt.plot(loss_l2_raw_full_epochs, color='green', marker='o',label='L2_RAW_FULL')
            plt.plot(loss_l2_raw_o_epochs, color='blue', marker='o',label='L2_RAW_O')
            plt.plot(loss_lpips_raw_full_epochs, color='magenta', marker='o',label='LPIPS_RAW_FULL')
            plt.plot(loss_lpips_raw_o_epochs, color='yellow', marker='o',label='LPIPS_RAW_O')
            plt.plot(loss_l2_sr_full_epochs, color='purple', marker='o', label='L2_SR_FULL')
            plt.plot(loss_l2_sr_o_epochs, color='black', marker='o', label='L2_SR_O')
            plt.plot(loss_blendw_epochs, color='purple', marker='x', label='LOSS_BLENDW')
            plt.plot(loss_blendw_area_epochs, color='black', marker='x', label='LOSS_BLENDW_AREA')
            plt.plot(loss_blendw_sparse_epochs, color='red', marker='x', label='LOSS_BLENDW_SPARSE')
            plt.plot(loss_depth_epochs, color='green', marker='x', label='LOSS_DEPTH_REG')
            plt.plot(loss_total_epochs, color='red', marker='o',label='Total')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.savefig(os.path.join(outdir, 'training_loss.jpg'))
            plt.close('all')

    if train_sr:
        if not train_raw:
            # load ckpt
            ckpt = torch.load(ckpt_path)
            compose_triplanes.load_state_dict(ckpt['compose_triplanes'])
            phi = ckpt['phi']
            outdir = os.path.join(outdir, 'train_sr')
        else:
            outdir = os.path.join(outdir, 'train_sr')

        for param in compose_triplanes.parameters():
            param.requires_grad = False
        for param in compose_triplanes.G.parameters():
            param.requires_grad = False
        for param in compose_triplanes.G.superresolution.parameters():
            param.requires_grad = True

        # build optimizers
        optimizer = torch.optim.Adam([
            {'params': compose_triplanes.G.superresolution.parameters(), 'lr': float(lr_sr)}
        ])
        
        compose_triplanes.eval()
        
        # build criterions
        if not use_mask:
            l2_fn = torch.nn.MSELoss(reduction='mean')
        else:
            l2_fn = torch.nn.MSELoss(reduction='sum')
        lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device).eval()   

        # start training
        global_iters = 0
        # lr_rampdown_length = 0.25
        # lr_rampup_length = 0.05
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        if save_intermediates:
            os.makedirs(os.path.join(outdir, 'intermediates'), exist_ok=True)

        loss_l2_sr_full_epochs = []
        loss_lpips_sr_full_epochs = []
        loss_total_epochs = []
        num_epochs = num_epochs_sr

        for epoch in range(num_epochs):

            loss_l2_sr_full_iters = []
            loss_lpips_sr_full_iters = []
            loss_total_iters = []

            for itr, data in enumerate(train_dataloader):
                if video_dataset.depth_maps is not None:
                    target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, _, _ = data
                else:
                    target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, _ = data
                target_images = target_images.to(device)
                target_images = F.interpolate(target_images, size=(G.img_resolution, G.img_resolution), mode='bilinear')
                camera_params = camera_params.to(device)
                w_plus_curr = w_plus_curr.to(device)
                phi_curr = phi_curr.to(device)

                if use_mask:
                    # masks_tensor = 1 - masks_tensor
                    masks_tensor = masks_tensor.to(device)
                    masks_tensor = F.interpolate(masks_tensor, size=(G.img_resolution, G.img_resolution), mode='nearest')

                if use_raw_rgb_loss:
                    target_images_downsampled = F.interpolate(target_images.clone(), size=(128, 128), mode='bilinear')
                    if use_mask:
                        mask_downsampled = F.interpolate(masks_tensor, size=(128, 128), mode='nearest')

                # forward pass
                out = compose_triplanes(w_plus_curr, camera_params, phi_curr, masks_tensor, return_raw_blendw=return_raw_blendw)
                
                synth_images_full = out['image_full']

                
                loss_dict = {}
                loss = 0.0

                if lamb_l2_sr_full > 0.0:
                    l2_sr_full = l2_fn(synth_images_full, target_images)/(target_images.shape[0] * target_images.shape[1] * target_images.shape[2] * target_images.shape[3])
                    loss += lamb_l2_sr_full * l2_sr_full
                    loss_dict["SR_L2_FULL"] = l2_sr_full.item()
                    loss_l2_sr_full_iters.append(loss_dict["SR_L2_FULL"])
                else:
                    loss_l2_sr_full_iters.append(0.0)
                
                if lamb_lpips_sr_full > 0.0:
                    lpips_sr_full = lpips_fn(synth_images_full, target_images).mean()
                    loss += lamb_lpips_sr_full * lpips_sr_full
                    loss_dict["SR_LPIPS_FULL"] = lpips_sr_full.item()
                    loss_lpips_sr_full_iters.append(loss_dict["SR_LPIPS_FULL"])
                else:
                    loss_lpips_sr_full_iters.append(0.0)

                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                print_txt = f'Epoch {epoch+1:>4d}/{num_epochs}, Itr {global_iters+1}: '
                for loss_key in loss_dict:
                    print_txt += loss_key + f' {loss_dict[loss_key]:<4.2f} ' 
                print_txt += f' loss {float(loss):<5.2f}'
                print(print_txt)
                            
                loss_total_iters.append(loss.item())

                # save intermediate result
                if save_intermediates and ((global_iters + 1) % vis_step == 0 or global_iters == 0):  # debug
                # if save_intermediates and data[-1][0] == '/home/yiranx/data/capvideo_simple/11986_16f/frames/frame0203.png':
                    with torch.no_grad():
                        out = compose_triplanes(w_plus_curr, camera_params, phi_curr, masks_tensor)
                        synth_image_full_sr = out['image_full']

                        synth_image_full_sr = (synth_image_full_sr + 1) * (255/2)
                        synth_image_full_sr = synth_image_full_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()


                        synth_image_f = compose_triplanes.G.synthesis(w_plus_curr, camera_params)['image']
                        synth_image_f = (synth_image_f + 1) * (255/2)
                        synth_image_f = synth_image_f.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                        
                        target_image = target_images[0:1]
                        target_image = (target_image + 1) * (255/2)
                        target_image = target_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                        # ood rgb, full composite rgb, target
                        output_image = np.concatenate((synth_image_f, synth_image_full_sr, target_image), axis=1)
                        imageio.imwrite(os.path.join(outdir, 'intermediates', 'step_{:05}.jpg'.format(global_iters+1) ), output_image)

                        torch.save({'w_plus': w_plus.cpu(),
                                    'phi': phi.cpu(),
                                    'compose_triplanes': compose_triplanes.state_dict(),
                                    }, f'{outdir}/triplanes.pt')

                global_iters += 1
            

            loss_l2_sr_full_epochs.append(np.mean(loss_l2_sr_full_iters))
            loss_lpips_sr_full_epochs.append(np.mean(loss_lpips_sr_full_iters))
            loss_total_epochs.append(np.mean(loss_total_iters))

            plt.figure()
            plt.plot(loss_l2_sr_full_epochs, color='green', marker='o',label='L2_SR_FULL')
            plt.plot(loss_lpips_sr_full_epochs, color='blue', marker='o',label='LPIPS_SR_FULL')
            plt.plot(loss_total_epochs, color='red', marker='o',label='TOTAL')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.savefig(os.path.join(outdir, 'training_loss.jpg'))
            plt.close('all')
    
    # visualize and save results
    os.makedirs(os.path.join(outdir, 'frames', 'projected'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_o'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_sr'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'projected_o_sr'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'targets'), exist_ok=True)
    eval_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Saving results...")
    f = open(os.path.join(outdir, 'target_frames.txt'), 'w')
    compose_triplanes.eval()
    
    # ckpt = torch.load('/fs/nexus-scratch/yiranx/codes/eg3d/eg3d/out/optim_multi_dynamic/rednose2_16f_maskeditem_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_prepose_noUpdatePose/outdomain_1017_Better4WeirdGrids/triplanes.pt')
    # compose_triplanes.load_state_dict(ckpt['compose_triplanes'])
    with torch.no_grad():
        for itr, data in enumerate(eval_dataloader):
            if video_dataset.depth_maps is not None:
                target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, _, img_path = data
            else:
                target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, img_path = data
            camera_params = camera_params.to(device)
            w_plus_curr = w_plus_curr.to(device)
            phi_curr = phi_curr.to(device)
            out = compose_triplanes(w_plus_curr, camera_params, phi_curr, masks_tensor)
            
            synth_image = out['image_raw_full']
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            target_images = (target_images + 1) * (255/2)
            target_images = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_o = out['image_raw_o']
            synth_image_o = (synth_image_o + 1) * (255/2)
            synth_image_o = synth_image_o.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_full_sr = out['image_full']
            synth_image_full_sr = (synth_image_full_sr + 1) * (255/2)
            synth_image_full_sr = synth_image_full_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_o_sr = out['image_o']
            synth_image_o_sr = (synth_image_o_sr + 1) * (255/2)
            synth_image_o_sr = synth_image_o_sr.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                

            PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'frames', 'projected', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_o, 'RGB').save(os.path.join(outdir, 'frames', 'projected_o', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_full_sr, 'RGB').save(os.path.join(outdir, 'frames', 'projected_sr', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(synth_image_o_sr, 'RGB').save(os.path.join(outdir, 'frames', 'projected_o_sr', 'proj_{0:05d}.png'.format(itr)))
            PIL.Image.fromarray(target_images, 'RGB').save(os.path.join(outdir, 'frames', 'targets', 'target_{0:05d}.png'.format(itr)))
            f.write(os.path.basename(img_path[0]) + '\n')

            # if itr == 0: # visualize canonical frame
            #     # synth_image = G.synthesis(ws_plus_cano_opt, camera_params_opt_curr)['image']
            #     synth_image = G.synthesis(ws_plus_cano_opt, camera_params)['image']

            #     synth_image = (synth_image + 1) * (255/2)
            #     synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            #     PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'frames', 'cano_{0:05d}.png'.format(itr)))
    f.close()

    torch.save({'w_plus': w_plus.cpu(),
                'phi': phi.cpu(),
                'compose_triplanes': compose_triplanes.state_dict(),
                }, f'{outdir}/triplanes.pt')

if __name__ == '__main__':
    torch.manual_seed(20221026)
    torch.cuda.manual_seed(20221026)
    np.random.seed(20221026)
    outdomain_inv()
