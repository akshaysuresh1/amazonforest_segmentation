"""
Definition of SegmentationDataset class
"""

from typing import List, Tuple, Callable
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from ..ops.tif_utils import load_tif_from_s3
from ..ops.scaling_utils import robust_scaling
from ..resources import s3_resource


class SegmentationDataset(Dataset):
    """
    Dataset object for binary semantic segmentation
    """

    def __init__(
        self,
        images_list: List[str],
        masks_list: List[str],
        s3_bucket: str,
        scaling_func: Callable = robust_scaling,
        transform: A.Compose | None = None,
    ) -> None:
        """
        Initialize a SegmentationDataset object.

        Args:
            images_list: List of image filenames (or object keys)
            masks_list: List of segmentation masks corresponding to images in "images_list"
            s3_bucket: Name of S3 bucket containing images and masks
            scaling_func: Scaling function to be applied to image (d: robust_scaling)
            transform: Augmentation pipeline
        """
        if len(images_list) != len(masks_list):
            raise ValueError("Unequal numbers of images and masks supplied.")

        self.images = images_list
        self.masks = masks_list
        self.s3_bucket = s3_bucket
        self.scaling_func = scaling_func
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves image-mask pair at specified index from dataset

        Returns:
            image: Multispectral image
            mask: Binary segmentation mask
        """
        # Read in image and segmentation mask as xarray datasets.
        # Image shape = (n_bands, n_y, n_x)
        image = load_tif_from_s3(s3_resource, self.s3_bucket, self.images[index])
        # Mask shape = (1, n_y, n_x)
        mask = load_tif_from_s3(s3_resource, self.s3_bucket, self.masks[index])

        if image.shape[1] != mask.shape[1]:
            raise ValueError(
                f"Index {index}: The image and the mask have unequal row counts."
            )

        if image.shape[2] != mask.shape[2]:
            raise ValueError(
                f"Index {index}: The image and the mask have unequal column counts."
            )

        # Convert image and mask to numpy arrays.
        image = image.to_numpy()  # shape = (n_bands, n_y, n_x)
        mask = mask.to_numpy().squeeze()  # shape = (n_y, n_x)

        # Shift color axis at index 0 to last index.
        image = np.moveaxis(image, 0, -1)  # shape = (n_y, n_x, n_bands)

        # Scale image appropriately for deep learning.
        image = self.scaling_func(image)

        if self.transform:
            # Apply transformation to both image and mask.
            transformed_products = self.transform(image=image, mask=mask)
            image = transformed_products.get("image")
            mask = transformed_products.get("mask")

        # Cast image and mask as torch tensors.
        # Image output shape = (n_bands, n_y, n_x)
        image = torch.from_numpy(image.copy().astype(np.float32)).permute((2, 0, 1))
        # Mask output shape = (1, n_y, n_x)
        mask = torch.from_numpy(mask.copy().astype(np.float32)).unsqueeze(0)

        return image, mask

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self.images)
