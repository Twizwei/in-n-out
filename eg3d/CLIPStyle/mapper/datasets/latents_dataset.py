from torch.utils.data import Dataset
import torch

class LatentsDataset(Dataset):

	def __init__(self, latents, opts, cam_params=None):
		self.latents = latents
		self.opts = opts
		self.cam_params = cam_params
		if cam_params is not None:
			self.intrinsics = self.cam_params['cam_intrinsics']
			self.extrinsics = self.cam_params['cam_params']

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		if self.cam_params is None:
			return self.latents[index]
		else:
			cam2world_pose = self.extrinsics[index]
			camera_params = torch.cat([cam2world_pose.reshape(16), self.intrinsics.reshape(9)], dim=-1)
			return self.latents[index], camera_params
