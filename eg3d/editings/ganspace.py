import torch


def edit(latents, latents_plus, pca, edit_directions):
    edit_latents = []
    edit_latents_plus = []
    for idx, latent_plus in enumerate(latents_plus):
        for pca_idx, start, end, strength in edit_directions:
            latent = latents[idx]
            delta = get_delta(pca, latent, pca_idx, strength)
            delta_padded = torch.zeros(latent_plus.shape).to('cuda')
            delta_padded[start:end] += delta.repeat(end - start, 1)
            edit_latents.append(latent + delta)
            edit_latents_plus.append(latent_plus + delta_padded)
    return torch.cat(edit_latents), torch.stack(edit_latents_plus)


def get_delta(pca, latent, idx, strength, device='cuda'):
    # pca: ganspace checkpoint. latent: (8, 512) w+
    w_centered = latent - torch.from_numpy(pca['lat_mean']).to(device)
    lat_comp = torch.from_numpy(pca['lat_comp']).to(device)
    lat_std = torch.from_numpy(pca['lat_stdev']).to(device)
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta