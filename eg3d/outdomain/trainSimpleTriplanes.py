"""
Only finetune triplanes, decoder (and camera parameters?)
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
@click.option('--target_path', help='Target frame path', type=str, required=True, metavar='DIR')
@click.option('--latents_path', help='latent code path', type=str, )
@click.option('--outdir', help='Where to save the output results', type=str, required=True, metavar='DIR')
@click.option('--num_epochs', help='Number of epochs', type=int, default=500, show_default=True)
@click.option('--lr', help='learning rate', type=float, metavar='float', default=1e-2, show_default=True)
@click.option('--weight_lpips', help='weight of lpips loss', type=float, metavar='float', default=0.5, show_default=True)
# @click.option('--weight_residual', help='weight of residual', type=float, metavar='float', default=1.0, show_default=True)
# @click.option('--weight_delta_norm', help='weight of delta_norm', type=float, metavar='float', default=2e-3, show_default=True)
# @click.option('--w_init_samples', type=int, help='Samples for initialization', default=2000)
@click.option('--batch_size', type=int, help='batch size', default=1)
@click.option('--save_intermediates', type=bool, help='if save intermediate results', default=True, show_default=True)
@click.option('--random_init', type=bool, help='if randomly initialize variables', default=True, show_default=True)
@click.option('--use_raw_rgb_loss', type=bool, help='if use raw rgb', default=False, show_default=True)
@click.option('--use_mask', type=bool, help='if use mask', default=False, show_default=True)
@click.option('--lamb_l2_raw_full', type=float, help='weight of l2 loss for raw rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_l2_raw_o', type=float, help='weight of l2 loss for out-of-distribution part', default=1.0, show_default=True)
@click.option('--lamb_lpips_raw_full', type=float, help='weight of lpips loss for raw rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_lpips_raw_o', type=float, help='weight of lpips loss for out-of-distribution part', default=1.0, show_default=True)
@click.option('--lamb_l2_sr_full', type=float, help='weight of l2 loss for SR rgb in full composition', default=1.0, show_default=True)
@click.option('--lamb_l2_sr_o', type=float, help='weight of l2 loss for SR rgb out-of-distribution part', default=1.0, show_default=True)
@click.option('--output_resolution', type=int, help='neural rendering output resolution', default=128, show_default=True)


def outdomain_inv(
        network_pkl,
        target_path,
        latents_path,
        outdir,
        num_epochs=500,
        lr=1e-3,
        weight_lpips=0.5,
        # weight_residual=1.0,
        # weight_delta_norm=2e-3,
        # w_init_samples=2000,
        batch_size=1,
        save_intermediates=True,
        random_init=True,
        use_raw_rgb_loss=False,
        use_mask=True,
        lamb_l2_raw_full=1.0,
        lamb_l2_raw_o=1.0,
        lamb_lpips_raw_full=1.0,
        lamb_lpips_raw_o=1.0,
        lamb_l2_sr_full=1.0,
        lamb_l2_sr_o=1.0,
        output_resolution=128,
    ):
    device = torch.device('cuda')

    # load initial latent codes (from other inversion method)
    latents = torch.load(latents_path)
    w_plus = latents['w_plus']
    
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
    

    triplaneField.triplane = triplaneField.triplane.requires_grad_(True)
    # build optimizers
    optimizer = torch.optim.Adam(triplaneField.parameters(), lr=lr)


    # build criterions
    if not use_mask:
        l2_fn = torch.nn.MSELoss(reduction='mean')
    else:
        l2_fn = torch.nn.MSELoss(reduction='sum')
    lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device).eval()   

    # start training
    global_iters = 0
    lr_rampdown_length = 0.25
    lr_rampup_length = 0.05
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    if save_intermediates:
        os.makedirs(os.path.join(outdir, 'intermediates'), exist_ok=True)

    loss_l2_raw_o_epochs = []
    loss_lpips_raw_o_epochs = []
    loss_total_epochs = []
    for epoch in range(num_epochs):
        # Learning rate schedule.
        # t = epoch / num_epochs
        # lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        # lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        # lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        # lr = lr * lr_ramp
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        loss_l2_raw_o_iters = []
        loss_lpips_raw_o_iters = []
        loss_total_iters = []

        for itr, data in enumerate(train_dataloader):
            target_images, masks_tensor, camera_params, w_plus_curr, phi_curr,_ = data
            target_images = target_images.to(device)
            target_images = F.interpolate(target_images, size=(triplaneField.neural_rendering_resolution, triplaneField.neural_rendering_resolution), mode='bilinear')
            camera_params = camera_params.to(device)
            w_plus_curr = w_plus_curr.to(device)
            phi_curr = phi_curr.to(device)

            if use_mask:
                # masks_tensor = 1 - masks_tensor
                masks_tensor = masks_tensor.to(device)
                masks_tensor = F.interpolate(masks_tensor, size=(triplaneField.neural_rendering_resolution, triplaneField.neural_rendering_resolution), mode='nearest')

            if use_raw_rgb_loss:
                target_images_downsampled = F.interpolate(target_images.clone(), size=(128, 128), mode='bilinear')
                if use_mask:
                    mask_downsampled = F.interpolate(masks_tensor, size=(128, 128), mode='nearest')

            # forward pass
            out = triplaneField(c=camera_params)

            thumb_rgb_o = out['image_raw'] 

            loss_dict = {}
            loss = 0.0

                
            if lamb_l2_raw_o > 0.0:
                l2_raw_o = l2_fn(thumb_rgb_o * (1 - mask_downsampled), target_images_downsampled * (1 - mask_downsampled))/(1 - mask_downsampled).sum()
                # l2_raw_o = l2_fn(thumb_rgb_o * (1 - mask_downsampled), target_images_downsampled * (1 - mask_downsampled))/((1 - mask_downsampled).sum() + 1)
                # l2_raw_o = l2_fn(thumb_rgb_o, target_images_downsampled)/(thumb_rgb_o.shape[0] * thumb_rgb_o.shape[1] * thumb_rgb_o.shape[2] * thumb_rgb_o.shape[3])
                loss += lamb_l2_raw_o * l2_raw_o
                loss_dict["RAW_L2_O"] = l2_raw_o.item()
                loss_l2_raw_o_iters.append(loss_dict["RAW_L2_O"])
            else:
                loss_l2_raw_o_iters.append(0.0)



            if lamb_lpips_raw_o > 0.0:
                lpips_raw_o = (lpips_fn(thumb_rgb_o, target_images_downsampled) * (1 - mask_downsampled)).sum()/(1 - mask_downsampled).sum()
                loss += lpips_raw_o * lpips_raw_o
                loss_dict["RAW_LPIPS_O"] = lpips_raw_o.item()
                loss_lpips_raw_o_iters.append(loss_dict["RAW_LPIPS_O"])
            else:
                loss_lpips_raw_o_iters.append(0.0)
 

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
            if save_intermediates and ((global_iters + 1) % 200 == 0 or global_iters == 0):  # debug
            # if save_intermediates and data[-1][0] == '/home/yiranx/data/capvideo_simple/11986_16f/frames/frame0203.png':
                with torch.no_grad():
                    out = triplaneField(c=camera_params)
                    synth_image_o = out['image_raw']
                    
                    synth_image_o = F.interpolate(synth_image_o, size=(128, 128), mode='area')
                    synth_image_o = (synth_image_o + 1) * (255/2)
                    synth_image_o = synth_image_o.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    
                    target_image = F.interpolate(target_images[0:1], size=(128, 128), mode='area')
                    target_image = (target_image + 1) * (255/2)
                    target_image = target_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                    # ood rgb, full composite rgb, target
                    output_image = np.concatenate((synth_image_o, target_image), axis=1)
                    imageio.imwrite(os.path.join(outdir, 'intermediates', 'step_{:05}.jpg'.format(global_iters+1) ), output_image)

                    torch.save({'w_plus': w_plus.cpu(),
                            'triplanes': triplaneField.state_dict(),
                            }, f'{outdir}/triplanes.pt')
            global_iters += 1
        
        loss_l2_raw_o_epochs.append(np.mean(loss_l2_raw_o_iters))
        loss_lpips_raw_o_epochs.append(np.mean(loss_lpips_raw_o_iters))
        loss_total_epochs.append(np.mean(loss_total_iters))

        plt.figure()
        plt.plot(loss_l2_raw_o_epochs, color='blue', marker='o',label='L2_RAW_O')
        plt.plot(loss_lpips_raw_o_epochs, color='yellow', marker='o',label='LPIPS_RAW_O')
        plt.plot(loss_total_epochs, color='red', marker='o',label='Total')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(outdir, 'training_loss.jpg'))
        plt.close('all')
    
    # visualize and save results
    os.makedirs(os.path.join(outdir, 'frames', 'projected_o'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'frames', 'targets'), exist_ok=True)
    eval_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Saving results...")
    f = open(os.path.join(outdir, 'target_frames.txt'), 'w')    

    with torch.no_grad():
        for itr, data in enumerate(eval_dataloader):
            target_images, masks_tensor, camera_params, w_plus_curr, phi_curr, img_path = data
            camera_params = camera_params.to(device)
            w_plus_curr = w_plus_curr.to(device)
            phi_curr = phi_curr.to(device)

            out = triplaneField(c=camera_params)
            
           
            target_images = (target_images + 1) * (255/2)
            target_images = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            synth_image_o = out['image_raw']
            synth_image_o = (synth_image_o + 1) * (255/2)
            synth_image_o = synth_image_o.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                

            PIL.Image.fromarray(synth_image_o, 'RGB').save(os.path.join(outdir, 'frames', 'projected_o', 'proj_{0:05d}.png'.format(itr)))
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
                'triplanes': triplaneField.state_dict(),
                }, f'{outdir}/triplanes.pt')

if __name__ == '__main__':
    outdomain_inv()