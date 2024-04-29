import os
import glob

from PIL import Image 
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import ToTensor, Normalize
import pdb

intrinsics = torch.tensor([[4.4652, 0.0000, 0.5000],
                            [0.0000, 4.4652, 0.5000],
                            [0.0000, 0.0000, 1.0000]])

# class MutilFrameDataset(Dataset):
#     def __init__(self, data_root, azim_opt=None, elev_opt=None, load_latents=False):
#         self.data_root = data_root
#         self.load_latents = load_latents
        
#         self.frame_dir = os.path.join(data_root, 'frames')
#         self.frame_list = sorted(glob.glob(os.path.join(self.frame_dir, '*png')))

#         self.mask_dir = os.path.join(data_root, 'masks')
#         self.mask_list = sorted(glob.glob(os.path.join(self.mask_dir, '*png')))
        
#         if os.path.exists(os.path.join(data_root, 'cam_params.pt')):
#             self.cam_params = torch.load(os.path.join(data_root, 'cam_params.pt'), map_location='cpu')  # dict_keys(['extrinsics', 'focals', 'near', 'far'])
#         else:
#             self.cam_params = None

#         if load_latents:
#             self.latent_z = torch.load(os.path.join(data_root, 'latent_z.pt'))
#             self.latent_w = torch.load(os.path.join(data_root, 'latent_w.pt'))

#         self.transforms = T.Compose(
#             [
#                 ToTensor(),
#                 Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#             ]
#         )

#         # assume they're sorted
#         self.azim_opt = azim_opt
#         self.elev_opt = elev_opt
    
#     def frame_truncation(self, num_frames):
#         """
#             Randomly choose num_frames frames.
#         """

#         print("Randomly choose {} frames left.".format(num_frames))
        
#         idx = sorted(random.choices(range(len(self.frame_list)), k=num_frames))
#         self.frame_list = list(np.array(self.frame_list)[idx])  # to fix: kinda inefficient
#         if 'extrinsics' in self.cam_params: 
#             self.cam_params['extrinsics'] = self.cam_params['extrinsics'][idx]
#         else:
#             self.cam_params['azim'] = self.cam_params['azim'][idx]
#             self.cam_params['elev'] = self.cam_params['elev'][idx]
#         self.cam_params['focals'] = self.cam_params['focals'][idx]
#         self.cam_params['near'] = self.cam_params['near'][idx]
#         self.cam_params['far'] = self.cam_params['far'][idx]
#         if self.load_latents:
#             self.latent_z = self.latent_z[idx]
#             self.latent_w = self.latent_w[idx]
        
#         if self.azim_opt is not None:
#             self.azim_opt = self.azim_opt[idx]
        
#         if self.elev_opt is not None:
#             self.elev_opt = self.elev_opt[idx]

#     def __len__(self):
#         return len(self.frame_list)
    
#     def __getitem__(self, idx):
#         # read image
#         img_path = self.frame_list[idx]
#         img = Image.open(img_path)

#         # transform image
#         img_tensor = self.transforms(img)

#         # read mask
#         if len(self.mask_list) > 0:
#             mask_tensor = np.array(Image.open(self.mask_list[idx]))
#             mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0).to(torch.float32)/255.
#             mask_tensor = mask_tensor.expand(3, mask_tensor.shape[1], mask_tensor.shape[2])
#         else:
#             mask_tensor = torch.ones(3, img_tensor.shape[-2], img_tensor.shape[-1]).to(torch.float32)

#         # assign camera parameters
#         if self.cam_params is not None:
#             if 'extrinsics' in self.cam_params:
#                 cam_extrinsics = self.cam_params['extrinsics'][idx]
#             else:
#                 cam_extrinsics = torch.cat([self.cam_params['azim'][idx], self.cam_params['elev'][idx]])
#             cam_focals = self.cam_params['focals'][idx]
#             cam_near = self.cam_params['near'][idx]
#             cam_far = self.cam_params['far'][idx]
#         else:
#             cam_extrinsics = torch.tensor(1.0) # dumpy data
#             cam_focals = sample_focals
#             cam_near = sample_near
#             cam_far = sample_far

#         # output optimized variables
#         if (self.azim_opt is not None) and (self.elev_opt is not None): 
#             azim_opt_curr = self.azim_opt[idx]
#             elev_opt_curr = self.elev_opt[idx]

#             return img_tensor, mask_tensor, cam_extrinsics, cam_focals, cam_near, cam_far, azim_opt_curr, elev_opt_curr, img_path

#         else:
#             return img_tensor, mask_tensor, cam_extrinsics, cam_focals, cam_near, cam_far, img_path


class MutilFrameDatasetDynamic(Dataset):
    def __init__(self, data_root, w=None, w_plus=None, yaw_opt=None, pitch_opt=None, use_pre_pose=False):
        self.data_root = data_root
        
        self.frame_dir = os.path.join(data_root, 'frames')
        self.frame_list = sorted(glob.glob(os.path.join(self.frame_dir, '*png')))
        # remove frame that ends with _mask.png
        self.frame_list = [x for x in self.frame_list if not x.endswith('_mask.png')]

        self.mask_dir = os.path.join(data_root, 'masks')
        self.mask_list = sorted(glob.glob(os.path.join(self.mask_dir, '*png')))
        
        if os.path.exists(os.path.join(data_root, 'cam_params.pt')):
            self.cam_params = torch.load(os.path.join(data_root, 'cam_params.pt'), map_location='cpu')  # dict_keys(['extrinsics', 'focals', 'near', 'far'])
        else:
            self.cam_params = None

        self.use_pre_pose = use_pre_pose
        if os.path.exists(os.path.join(data_root, 'dataset.json')):
            import json
            with open(os.path.join(data_root, 'dataset.json'), 'r') as f:
                self.cam_params_json = json.load(f)['labels']
            f.close()
        else:
            self.cam_params_json = None

        self.transforms = T.Compose(
            [
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # assume they're sorted
        self.yaw_opt = yaw_opt
        self.pitch_opt = pitch_opt
        self.w = w
        self.w_plus = w_plus
    
    def frame_truncation(self, num_frames):
        """
            Randomly choose num_frames frames.
        """

        # print("Randomly choose {} frames left.".format(num_frames))
        
        # idx = sorted(random.choices(range(len(self.frame_list)), k=num_frames))
        # self.frame_list = list(np.array(self.frame_list)[idx])  # to fix: kinda inefficient
        # if 'extrinsics' in self.cam_params: 
        #     self.cam_params['extrinsics'] = self.cam_params['extrinsics'][idx]
        # else:
        #     self.cam_params['azim'] = self.cam_params['azim'][idx]
        #     self.cam_params['elev'] = self.cam_params['elev'][idx]
        # self.cam_params['focals'] = self.cam_params['focals'][idx]
        # self.cam_params['near'] = self.cam_params['near'][idx]
        # self.cam_params['far'] = self.cam_params['far'][idx]
        # if self.load_latents:
        #     self.latent_z = self.latent_z[idx]
        #     self.latent_w = self.latent_w[idx]
        
        # if self.azim_opt is not None:
        #     self.azim_opt = self.azim_opt[idx]
        
        # if self.elev_opt is not None:
        #     self.elev_opt = self.elev_opt[idx]
        raise NotImplementedError

    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        # read image
        img_path = self.frame_list[idx]
        img = Image.open(img_path)

        # transform image
        img_tensor = self.transforms(img)

        # read mask
        if len(self.mask_list) > 0:
            assert os.path.basename(self.mask_list[idx]) == os.path.basename(img_path), "image and mask does not match!"
            mask_tensor = np.array(Image.open(self.mask_list[idx]))
            mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0).to(torch.float32)/255.
            mask_tensor = mask_tensor.expand(3, mask_tensor.shape[1], mask_tensor.shape[2])

            # mask_tensor = torch.from_numpy(mask_tensor).permute(2,0,1).to(torch.float32)/255.
            
        else:
            mask_tensor = torch.ones(3, img_tensor.shape[-2], img_tensor.shape[-1]).to(torch.float32)

        # assign camera parameters
        if self.cam_params is not None:
            if isinstance(self.cam_params, dict) and 'proj_mat' in self.cam_params:
                cam_extrinsics = self.cam_params['proj_mat'][idx]
            else:
                # cam_extrinsics = torch.cat([self.cam_params['yaw'][idx], self.cam_params['pitch'][idx]])
                cam_extrinsics = self.cam_params[idx]
                # cam_intrinsics = self.cam_params['intrinsics'][idx]
        else:
            cam_extrinsics = torch.tensor(1.0) # dumpy data
            cam_intrinsics = intrinsics
        
        if self.use_pre_pose and self.cam_params_json is not None:
            cam_param_curr = self.cam_params_json[idx]
            if isinstance(cam_param_curr, torch.Tensor):
                cam_param = cam_param_curr
            else:
                cam_param = torch.tensor(cam_param_curr[1])

        # output optimized variables
        if (self.yaw_opt is not None) and (self.pitch_opt is not None) and (self.w_plus is not None): 
            yaw_opt_curr = self.yaw_opt[idx]
            pitch_opt_curr = self.pitch_opt[idx]

            w_plus_opt_curr = self.w_plus[idx]

            return img_tensor, mask_tensor, cam_extrinsics, yaw_opt_curr, pitch_opt_curr, w_plus_opt_curr, img_path
        elif self.use_pre_pose and self.cam_params_json is not None and (self.w_plus is not None):
            w_plus_opt_curr = self.w_plus[idx]
            return img_tensor, mask_tensor, cam_param, w_plus_opt_curr, img_path
        elif  self.use_pre_pose and self.cam_params_json is not None and (self.w_plus is None):
            return img_tensor, mask_tensor, cam_param, img_path
        # else:
        #     yaw_opt_curr = self.yaw_opt[idx]
        #     pitch_opt_curr = self.pitch_opt[idx]
        #     return img_tensor, mask_tensor, cam_extrinsics, cam_intrinsics, yaw_opt_curr, pitch_opt_curr, img_path
        elif self.w_plus is not None:
            w_plus_opt_curr = self.w_plus[idx]
            return img_tensor, mask_tensor, cam_extrinsics, w_plus_opt_curr, img_path
        else:
            return img_tensor, mask_tensor, img_path