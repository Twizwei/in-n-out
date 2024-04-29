"""
Loss functions for out-of-domain rendering.
"""

import torch
import pdb
def compute_blendw_loss(coarse_blendw, fine_blendw, clip_threshold=1e-19, skewness=1.0, use_lap=False):
  """
  Compute the blendw loss based on entropy or lap
  skewness is used to control the skew of entropy loss. A value larger than 1.0 causes the peak to skew towards 1
  https://github.com/d2nerf/d2nerf/blob/main/hypernerf/training.py#L178 
  Args:
  coarse_blendw, fine_blendw: [N, H*W, pts_per_ray, 1]
  """
  # coarse_blendw = coarse_blendw.view(1, -1)
  # fine_blendw = fine_blendw.view(1, -1)
  blendw = torch.cat([coarse_blendw, fine_blendw], -2).squeeze(-1)
  blendw = torch.clamp(blendw ** skewness, clip_threshold, 1-clip_threshold)
  rev_blendw = torch.clamp(1-blendw, clip_threshold) # a_max behaving weird with small clip threshold
  entropy = - (blendw * torch.log(blendw) + rev_blendw*torch.log(rev_blendw))
  lap = torch.exp(-blendw) + torch.exp(-rev_blendw)
  lap = -torch.log(torch.clamp(lap, clip_threshold))

  if use_lap:
    return lap
  else:
    return entropy

def compute_blendw_area_loss(coarse_blendw, fine_blendw):
  """
  Compute loss that encourage blendw to stay concentrated on a ray
  https://github.com/d2nerf/d2nerf/blob/main/hypernerf/training.py#L223 
  """
  loss = 0.
  for blendw in [coarse_blendw, fine_blendw]:
    area_loss = blendw.max(dim=-2)[0] ** 2
    # area_loss = torch.max(blendw, axis=-1) ** 2 # mask * torch.log(torch.sum(blendw, -1, keepdims=True)) #
    loss += area_loss.mean()

  return loss / 2.

def compute_sparse_loss(coarse_blendw, fine_blendw, mask):
  blendw = torch.cat([coarse_blendw, fine_blendw], -2).squeeze(-1)

  return (torch.abs(blendw) * mask).sum()/mask.sum()/blendw.shape[-1]