# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialShift(nn.Module):
    def __init__(self, net, n_segment=3, percent=3):
        super(SpatialShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.percent = percent

        print('=> Using percent: {}'.format(self.percent))

    def forward(self, x):
        x = self.shift(x, self.n_segment, percent=self.percent)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment=3, percent=0.2):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment

        x = x.view(n_batch, n_segment, c, h, w)

        grid_x, grid_y = np.meshgrid(np.arange(0, h), np.arange(0, w))
        
        cnt = int(h * w * 0.2)
        choice_x = np.random.choice(h, cnt, replace=True)
        choice_y = np.random.choice(w, cnt, replace=True)
        
        x[:, 0, :, choice_x[: cnt//2], choice_y[: cnt//2]], \
            x[:, 1, :, choice_x[: cnt//2], choice_y[: cnt//2]] = x[:, 1, :, choice_x[: cnt//2], choice_y[: cnt//2]], \
                                                                                                                    x[:, 0, :, choice_x[: cnt//2], choice_y[: cnt//2]]
        
        x[:, 1, :, choice_x[cnt//2: ], choice_y[cnt//2: ]], \
            x[:, 2, :, choice_x[cnt//2: ], choice_y[cnt//2: ]] = x[:, 2, :, choice_x[cnt//2: ], choice_y[cnt//2: ]], \
                                                                                                                    x[:, 1, :, choice_x[cnt//2: ], choice_y[cnt//2: ]]

        return x.view(nt, c, h, w)

    

def make_spatial_shift(net, n_segment, percent=0.2, place='blockres'):
    """
    n_segment: number of frames
    """
    n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    if place == 'block':
        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks'.format(len(blocks)))
            for i, b in enumerate(blocks):
                blocks[i] = SpatialShift(b, n_segment=this_segment, percent=percent)
            return nn.Sequential(*(blocks))

        net.layer1 = make_block_temporal(net.encoder.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.encoder.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.encoder.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.encoder.layer4, n_segment_list[3])

    elif 'blockres' in place:
        n_round = 1
        if len(list(net.encoder.layer3.children())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = SpatialShift(b.conv1, n_segment=this_segment, percent=percent)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.encoder.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.encoder.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.encoder.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.encoder.layer4, n_segment_list[3])
