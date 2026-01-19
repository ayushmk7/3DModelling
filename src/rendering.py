import torch
import torch.nn.functional as F
import numpy as np

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, depth_map, acc_map, weights

def render_rays(rays_o, rays_d, near, far, N_samples, network, network_query_fn, perturb=0., raw_noise_std=0., white_bkgd=False):
    z_vals = torch.linspace(near, far, N_samples)
    if perturb > 0.:
        m_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([m_mid, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], m_mid], -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    raw = network_query_fn(pts, rays_d, network)
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    return rgb_map, depth_map, acc_map
