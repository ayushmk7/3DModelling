import torch
import torch.optim as optim
import numpy as np
from src.model import NeRF
from src.encoding import PositionalEncoding
from src.rendering import render_rays, get_rays
from src.dataset import NeRFDataset

def train_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = NeRFDataset(args.basedir, split='train')
    H, W, focal = dataset.H, dataset.W, dataset.focal
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    embed_fn, input_ch = PositionalEncoding(args.multires), 63 
    embeddirs_fn, input_ch_views = PositionalEncoding(args.multires_views), 27
    
    model = NeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views, output_ch=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lrate)

    N_rand = args.N_rand
    
    def network_query_fn(pts, viewdirs, network):
        return network(torch.cat([pts, viewdirs], -1)) 

    for i in range(args.N_iters):
        img_i = np.random.choice(len(dataset))
        target = dataset.imgs[img_i]
        pose = dataset.poses[img_i]
        target = torch.Tensor(target).to(device)
        pose = torch.Tensor(pose).to(device)

        rays_o, rays_d = get_rays(H, W, K, pose)
        
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)
        coords = torch.reshape(coords, [-1,2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()
        
        rays_o = rays_o[select_coords[:,0], select_coords[:,1]]
        rays_d = rays_d[select_coords[:,0], select_coords[:,1]]
        target_s = target[select_coords[:,0], select_coords[:,1]]

        rgb, _, _ = render_rays(rays_o, rays_d, args.near, args.far, args.N_samples, model, network_query_fn)

        loss = torch.mean((rgb - target_s)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
