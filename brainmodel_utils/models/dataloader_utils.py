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


class DictArrayDataset(data.Dataset):
    """
    General dataset constructor using a dictionary of tensors.
    Arguments:
        dict_array : dictionary of numpy arrays
        t           : torchvision.transforms instance
    """

    def __init__(self, dict_array, t=None):
        assert isinstance(dict_array, dict)

        self.transforms = t
        self.dict_array = dict_array
        for k_idx, k in enumerate(self.dict_array.keys()):
            if k_idx == 0:
                self.n_elements = self.dict_array[k].shape[0]
            else:
                assert self.dict_array[k].shape[0] == self.n_elements

    def __getitem__(self, index):
        inputs = dict()
        for k in self.dict_array.keys():
            inputs[k] = self.dict_array[k][index]
            if self.transforms is not None:
                inputs[k] = self.transforms(inputs[k])
        return inputs

    def __len__(self):
        return self.n_elements

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
    if not isinstance(image_transforms, list):
        image_transforms = [image_transforms]
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

def get_dict_array_dataloader(
    dict_array,
    array_transforms=None,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
):
    """
    Inputs:
        dict_array   : dictionary of numpy arrays
        array_transforms : either a list of or a single (torchvision.transforms) instance for transformations
    Outputs:
        dataloader  : (torch.utils.data.DataLoader) for the image array
    """

    t = array_transforms
    if (array_transforms is not None) and isinstance(array_transforms, list):
        t = transforms.Compose(array_transforms)

    dataset = DictArrayDataset(
        dict_array=dict_array, t=t
    )
    dataloader = _acquire_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader
