import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception = False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        '''
        super(GeneratorResnet, self).__init__()
        # Input_size = 3, n, n
        self.inception = inception
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)


        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        return (torch.tanh(x) + 1) / 2 # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual

class SineAttack:
    """
    Sine Attack: Parameterizes perturbations using sinusoidal bases
    Based on: "Generative Transfer Attack" (Qin et al., 2023)
    """
    def __init__(self,
                 image_shape=(3, 32, 32),
                 epsilon=8/255,
                 num_freq=10,
                 amps = None,
                 freq_h = None,
                 freq_w = None,
                 phase_h = None,
                 phase_w = None,
                 device='cuda'):
        """
        Args:
            image_shape: (C, H, W) shape of input images
            epsilon: L-infinity bound for perturbations
            num_freq: Number of sinusoidal frequencies per channel
            device: 'cuda' or 'cpu'
        """
        self.image_shape = image_shape
        self.epsilon = epsilon
        self.num_freq = num_freq
        self.device = device

        C, H, W = image_shape

        # Learnable parameters for sine waves
        # Amplitudes: controls strength of each frequency component
        self.amplitudes = nn.Parameter(
            torch.randn(C, num_freq, device=device) * 0.2
        )

        # Frequencies: controls periodicity in spatial dimensions
        self.freq_h = freq_h if freq_h is not None  else nn.Parameter(
            torch.randn(C, num_freq, device=device) * 0.2
        )
        self.freq_w = freq_w if freq_w is not None  else nn.Parameter(
            torch.randn(C, num_freq, device=device) * 0.2
        )

        # Phases: controls spatial offset
        self.phase_h = phase_h if phase_h is not None  else nn.Parameter(
            torch.randn(C, num_freq, device=device) * 0.2
        )
        self.phase_w = phase_w if phase_w is not None  else nn.Parameter(
            torch.randn(C, num_freq, device=device) * 0.2
        )

        # Create spatial coordinate grids
        self.grid_h = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        self.grid_w = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)

    def generate_perturbation(self, batch_size=1):
        """
        Generate perturbation using current sine parameters

        Returns:
            delta: (B, C, H, W) perturbation tensor
        """
        C, H, W = self.image_shape

        # Initialize perturbation
        delta = torch.zeros(batch_size, C, H, W, device=self.device)

        # Sum over all frequency components
        for c in range(C):
            for f in range(self.num_freq):
                # Compute sine wave pattern
                pattern = (
                    torch.sin(2 * np.pi * self.freq_h[c, f] * self.grid_h + self.phase_h[c, f]) *
                    torch.sin(2 * np.pi * self.freq_w[c, f] * self.grid_w + self.phase_w[c, f])
                )

                # Add weighted pattern to perturbation
                delta[:, c:c+1, :, :] += self.amplitudes[c, f] * pattern

        # Clip to epsilon ball
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)

        return delta

    def parameters(self):
        """Return all learnable parameters"""
        return [self.amplitudes, self.freq_h, self.freq_w,
                self.phase_h, self.phase_w]
