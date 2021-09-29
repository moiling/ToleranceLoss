# Adapted from https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import math
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, alpha):
        alpha = np.array(alpha) / 255.
        # Adobe 1K
        fg_width = np.random.randint(10, 30)
        bg_width = np.random.randint(10, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        return Image.fromarray(trimap.astype(np.uint8))


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [padw, padh, 0, 0], fill=fill)
    return img


def resize_if_smaller(img, size):
    min_size = min(img.size)
    if min_size < size:
        rate = size / float(min_size)
        h, w = math.ceil(rate * img.size[0]), math.ceil(rate * img.size[1])
        img = F.resize(img, [w, h])
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class ResizeIfBiggerThan(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        for idx, image in enumerate(images):
            max_size = max(image.size)
            if max_size > self.size:
                rate = self.size / float(max_size)
                h, w = math.ceil(rate * image.size[0]), math.ceil(rate * image.size[1])
                images[idx] = F.resize(image, [w, h])
        return images


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, images):
        if random.random() < self.flip_prob:
            for idx, image in enumerate(images):
                images[idx] = F.hflip(image)
        return images


class Reduce(object):
    def __init__(self, reduce_rate=2):
        self.reduce_rate = reduce_rate

    def __call__(self, images):
        for idx, image in enumerate(images):
            w = round(image.size[0] / self.reduce_rate)
            h = round(image.size[1] / self.reduce_rate)
            images[idx] = F.resize(image, [w, h])
        return images


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        for idx, image in enumerate(images):
            images[idx] = F.resize(image, self.size)
        return images


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, images):
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])
        for idx, image in enumerate(images):
            images[idx] = F.rotate(image, angle)
        return images


class RandomCrop(object):
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, images):
        # image, trimap, matte should use the same params.
        size = np.random.choice(self.sizes)
        image = resize_if_smaller(images[0], size)
        crop_params = T.RandomCrop.get_params(image, (size, size))
        for idx, image in enumerate(images):
            images[idx] = F.crop(image, *crop_params)

        return images


class ToTensor(object):
    def __call__(self, images):
        for idx, image in enumerate(images):
            # convert numpy and PIL to tensor.
            images[idx] = F.to_tensor(image)
        return images


class Normalize(object):
    # call after to_tensor
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for idx, image in enumerate(images):
            if image.shape[0] == 3:
                images[idx] = F.normalize(image, mean=self.mean, std=self.std)
        return images
