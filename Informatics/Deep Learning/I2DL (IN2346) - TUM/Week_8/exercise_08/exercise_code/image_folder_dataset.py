"""
Definition of ImageFolderDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import torch
from .base_dataset import Dataset


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""

    def __init__(self, *args,
                 root=None,
                 images=None,
                 labels=None,
                 transform=None,
                 download_url=" https://vision.in.tum.de/webshare/g/i2dl/mnist.zip",
                 **kwargs):
        super().__init__(*args,
                         download_url=download_url,
                         root=root,
                         **kwargs)
        print(download_url)
        self.images = torch.load(os.path.join(root, images))
        if labels is not None:
            self.labels = torch.load(os.path.join(root, labels))
        else:
            self.labels = None
        # transform function that we will apply later for data preprocessing
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image


class RescaleTransform:
    """Transform class to rescale images to a given range"""

    def __init__(self, range_=(0, 1), old_range=(0, 255)):
        """
        :param range_: Value range to which images should be rescaled
        :param old_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = range_[0]
        self.max = range_[1]
        self._data_min = old_range[0]
        self._data_max = old_range[1]

    def __call__(self, images):
        images = images - self._data_min  # normalize to (0, data_max-data_min)
        images /= (self._data_max - self._data_min)  # normalize to (0, 1)
        images *= (self.max - self.min)  # norm to (0, target_max-target_min)
        images += self.min  # normalize to (target_min, target_max)

        return images


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """

    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        images = (images - self.mean) / self.std
        return images


class FlattenTransform:
    """Transform class that reshapes an image into a 1D array"""

    def __call__(self, image):
        return image.flatten()


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images


class RandomHorizontalFlip:
    """
    Transform class that flips an image horizontically randomly with a given probability.
    """

    def __init__(self, prob=0.5):
        """
        :param prob: Probability of the image being flipped
        """
        self.p = prob

    def __call__(self, image):
        rand = random.uniform(0, 1)
        if rand < self.p:
            image = np.flip(image, 1)
        return image


class SaltPepper:
    def __init__(self, grains=64, prob=0.5):
        self.grains = grains
        self.prob = prob

    def __call__(self, image):
        if random.uniform(0, 1) < self.prob:
            rows = np.random.randint(0, image.shape[0], size=self.grains)
            cols = np.random.randint(0, image.shape[1], size=self.grains)

            g = self.grains // 2
            image[rows[:g], cols[:g]] = 255
            image[rows[g:], cols[g:]] = 0
        return image


class RandomRoll:
    def __init__(self, dx=(-8, 8), dy=(-8, 8), prob=0.5):
        self.dx = dx
        self.dy = dy
        self.prob = prob

    def __call__(self, image):
        if random.uniform(0, 1) < self.prob:
            x = random.randint(self.dx[0], self.dx[1])
            y = random.randint(self.dy[0], self.dy[1])
            return np.roll(image, (x, y), axis=(0, 1))
        return image