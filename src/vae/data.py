from pathlib import Path

from PIL import Image
import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAHQ(Dataset):
    def __init__(self, root_data: Path, height: int = 1024, width: int = 1024) -> None:
        """
            [CelebAHQ](https://github.com/switchablenorms/CelebAMask-HQ)
        """
        super().__init__()
        self.root_data = root_data
        self.dir_images = root_data.joinpath("CelebA-HQ-img")
        self.paths_images = [f for f in self.dir_images.iterdir()]
        self.n = len(self.paths_images)

        self.height = height
        self.width = width
        self.resize = transforms.Resize((height, width))

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        path_image = self.paths_images[index]
        image = self.resize(Image.open(path_image))
        image = torch.tensor(numpy.array(image)/255)[..., :3]
        return image
