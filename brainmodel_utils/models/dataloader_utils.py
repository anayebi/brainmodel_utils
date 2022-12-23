import numpy as np
from torchvision import transforms
from torch.utils import data


class ArrayDataset(data.Dataset):
    """
    General dataset constructor using an array of images.
    Arguments:
        image_array : numpy array of shape (N, H, W, 3)
        t           : torchvision.transforms instance
    """

    def __init__(self, image_array, t=None):
        assert image_array.ndim == 4
        assert t is not None

        self.transforms = t
        self.image_array = image_array
        self.n_images = image_array.shape[0]

    def __getitem__(self, index):
        inputs = self.transforms(self.image_array[index, :, :, :])
        return inputs

    def __len__(self):
        return self.n_images


def _acquire_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory=True):
    assert isinstance(dataset, data.Dataset)
    loader = data.DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return loader


def get_image_array_dataloader(
    image_array,
    image_transforms=None,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
):
    """
    Inputs: 
        image_array   : (numpy.ndarray) (N, H, W, 3)
        image_transforms : (torchvision.transforms) instance for image transformations
    Outputs:
        dataloader  : (torch.utils.data.DataLoader) for the image array
    """

    if image_transforms is None:
        # normalize between 0 and 1 at a minimum
        image_transforms = [transforms.ToTensor()]
    image_transforms = [transforms.ToPILImage()] + image_transforms

    dataset = ArrayDataset(
        image_array=image_array, t=transforms.Compose(image_transforms)
    )
    dataloader = _acquire_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader
