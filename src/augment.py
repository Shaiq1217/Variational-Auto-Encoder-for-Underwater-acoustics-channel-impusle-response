import torch
import numpy as np
import random

class CIRAugment:
    def __init__(self, noise_std=0.01, shift_max=5, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.scale_range = scale_range

    def __call__(self, sample):
        # sample: Tensor of shape (2, 49, 49)
        real, imag = sample[0], sample[1]

        # Add Gaussian noise
        real += torch.randn_like(real) * self.noise_std
        imag += torch.randn_like(imag) * self.noise_std

        # Random amplitude scaling
        scale = random.uniform(*self.scale_range)
        real *= scale
        imag *= scale

        # Random time shift along the taps axis (axis=1)
        shift = random.randint(-self.shift_max, self.shift_max)
        real = torch.roll(real, shifts=shift, dims=1)
        imag = torch.roll(imag, shifts=shift, dims=1)

        return torch.stack([real, imag])
