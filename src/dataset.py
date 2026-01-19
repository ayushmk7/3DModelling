import torch
import numpy as np
import json
import os
from PIL import Image

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, basedir, split='train', half_res=False):
        self.basedir = basedir
        self.split = split
        self.half_res = half_res
        self.imgs = []
        self.poses = []
        self.meta = None

        with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
            self.meta = json.load(fp)

        for frame in self.meta['frames']:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            self.imgs.append(np.array(Image.open(fname)))
            self.poses.append(np.array(frame['transform_matrix']))

        self.imgs = (np.array(self.imgs) / 255.).astype(np.float32)
        self.poses = np.array(self.poses).astype(np.float32)
        
        self.H, self.W = self.imgs[0].shape[:2]
        self.focal = 0.5 * self.W / np.tan(0.5 * self.meta['camera_angle_x'])

        if half_res:
            self.H = self.H // 2
            self.W = self.W // 2
            self.focal = self.focal / 2.

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.poses[idx]
