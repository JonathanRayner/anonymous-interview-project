#!/usr/bin/env python3
from pathlib import Path
from data import TripletFaceDataset
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Pool(nn.Module):
    """
    See e.g. https://discuss.pytorch.org/t/how-do-i-create-an-l2-pooling-2d-layer/105562/4
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs["divisor_override"] = 1  # div 1 instead of kernel size
        self.pool = nn.AvgPool2d(*args, **kwargs)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.pool(x ** 2))


class InceptionBlock(nn.Module):
    """
    Based on the FaceNet paper (https://arxiv.org/pdf/1503.03832.pdf)
    and the original Inception paper (https://arxiv.org/pdf/1409.4842.pdf).

    We also update to some ideas from Inception v2 to save on compute,
    see Figure 5 of https://arxiv.org/pdf/1512.00567v3.pdf
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        # some ratios roughly based on the FaceNet paper
        pool_branch_ch = out_ch // 8
        small_branch_ch = out_ch // 4
        medium_branch_ch = out_ch // 2
        large_branch_ch = out_ch // 8

        total_branch_ch = pool_branch_ch + small_branch_ch + medium_branch_ch +large_branch_ch
        assert total_branch_ch == out_ch, f"InceptionBlock created {total_branch_ch} when {out_ch} expected"

        self.pooling_branch = nn.Sequential(L2Pool(3, padding=1, stride=1),
                                            nn.Conv2d(in_channels=in_ch,
                                                      out_channels=pool_branch_ch,
                                                      kernel_size=1,
                                                      padding="same"),
                                            nn.ReLU())

        self.small_branch = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                                    out_channels=small_branch_ch,
                                                    kernel_size=1,
                                                    padding="same"),
                                          nn.ReLU())

        self.medium_branch = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                                     out_channels=medium_branch_ch // 2,
                                                     kernel_size=1,
                                                     padding="same"),
                                           nn.ReLU(),
                                           nn.Conv2d(in_channels=medium_branch_ch // 2,
                                                     out_channels=medium_branch_ch,
                                                     kernel_size=3,
                                                     padding="same"),
                                           nn.ReLU())

        self.large_branch = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                                    out_channels=large_branch_ch // 2,
                                                    kernel_size=1,
                                                    padding="same"),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=large_branch_ch // 2,
                                                    out_channels=large_branch_ch,
                                                    kernel_size=3,
                                                    padding="same"),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=large_branch_ch,
                                                    out_channels=large_branch_ch,
                                                    kernel_size=3,
                                                    padding="same"),
                                          nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooling_branch = self.pooling_branch(x)
        small_branch = self.small_branch(x)
        medium_branch = self.medium_branch(x)
        large_branch = self.large_branch(x)

        print(pooling_branch.shape, small_branch.shape, medium_branch.shape, large_branch.shape)
        
        return torch.cat([pooling_branch,
                          small_branch,
                          medium_branch,
                          large_branch],
                         dim=1)

class ConvNet(nn.Module):
    """
    ~ 3 million params
    Can be scaled up a lot by adding more channels,
    and/or more inception blocks.
    """
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2), # 124 x 124
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, 3, stride=2), # 61 x 61
                                    nn.ReLU(),
                                    InceptionBlock(64, 64),
                                    InceptionBlock(64, 64),
                                    nn.Conv2d(64, 128, 3, stride=2), # 30 x 30
                                    nn.ReLU(),
                                    nn.Conv2d(128, 256, 3, stride=2), # 14 x 14
                                    nn.ReLU(),
                                    InceptionBlock(256, 256),
                                    InceptionBlock(256, 256),
                                    nn.Conv2d(256, 512, 3, stride=2), # 6 x 6
                                    nn.ReLU(),
                                    InceptionBlock(512, 512),
                                    InceptionBlock(512, 512),
                                    nn.AvgPool2d(6),
                                    nn.Flatten(),
                                    nn.Linear(512, 128))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def triplet_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Triplet loss e.g. eq. 1-4 of https://arxiv.org/pdf/1503.03832.pdf

    Expects the first n // 3 batch elements to be anchors,
    the next n // 3 to be positive examples,
    the next n // 3 to be negative examples

    Args:
        batch of anchor, positive, and negative predictions
        
    Returns:
        batch of loss predictions
    """
    anchor, positive, negative = torch.chunk(3)

    positive_distance = ((anchor - positive) ** 2).mean(dim=(1))
    negative_distance = ((anchor - negative) ** 2).mean(dim=(1))

    


    


    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path_train_set', type=Path)
    parser.add_argument('path_test_set', type=Path)
    args = parser.parse_args()

    # Implement execution here
    model = ConvNet()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))
