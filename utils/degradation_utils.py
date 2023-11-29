import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np

from utils.image_utils import crop_img


class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch, clean_patch

    # def _add_gaussian_noise(self, clean_patch, sigma):
    #     sigma_list = [15, 25, 50]
    #     noisy_image = clean_patch.copy()
    #     h, w, _ = clean_patch.shape
    #     div_w = w // 3
    #     for i in range(3):
    #         start_w = i * div_w
    #         end_w = (i + 1) * div_w
    #         patch = noisy_image[:, start_w:end_w, :]
    #         sigma = sigma_list[i]
    #         noise = np.random.randn(*patch.shape)
    #         noise_patch = np.clip(patch + noise * sigma, 0, 255).astype(np.uint8)
    #         noisy_image[:, start_w:end_w, :] = noise_patch
    #     return clean_patch, noisy_image


    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1
