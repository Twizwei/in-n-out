import torch
import matplotlib

def render_depth(values, colormap_name="magma_r"):
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return colors

def calibrate_disparity(depth_est_orig, depth_map):
    """
    Calibrate the estimated depth map to depth map from renderer.
    Input:
        depth_est_orig: estimated disparity map from e.g. MiDaS
        depth_map: depth map from NeRF
    """
    with torch.no_grad():        
        disparity_est_orig = 1 / (depth_est_orig + 1e-8)
        W, H = depth_map.shape[1], depth_map.shape[0]
        depth_nerf = depth_map.cpu().view(H,W)
        assert disparity_est_orig.shape[0] == H and disparity_est_orig.shape[1] == W, "disparity_est_orig.shape != depth_map.shape, disparity_est_orig.shape: {}, depth_map.shape: {}".format(disparity_est_orig.shape, depth_map.shape)

        disparity_nerf = 1 / (depth_nerf + 1e-8)
        
        # if disparity_est_orig.shape[0] != H or disparity_est_orig.shape[1] != W:
        #     trans = torch.nn.Sequential(torchvision.transforms.Resize((H,W)))
        #     dp = torch.Tensor(disparity_est_orig).unsqueeze(0)
        #     dp_ = torch.cat([dp,dp,dp],dim=0)
        #     dp = trans(dp_)
        #     disparity_est_orig = dp[0,:,:].numpy()

        # Shift and normalize estimated disparity
        disparity_est = (disparity_est_orig-disparity_est_orig.min())/(disparity_est_orig.max()-disparity_est_orig.min())*(1 - 0.01) + 0.01

        # Scale
        median_nerf = disparity_nerf.median()
        median_est = torch.median(disparity_est)
        scale_nerf = (abs(disparity_nerf - median_nerf)).mean()
        scale_est = torch.mean(abs(disparity_est - median_est))

        # MiDaS formula
        disparity_aligned = (scale_nerf / scale_est) * (disparity_est - median_est) + median_nerf
        
        depth_aligned = 1 / (1e-8 + disparity_aligned)
        # depth_est = 1 / (1e-8 + disparity_est_orig)


    return depth_aligned.squeeze()