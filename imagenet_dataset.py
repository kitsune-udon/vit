import os

from PIL import Image
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, dataset_root='.', split='train', transform=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        assert split in ['train', 'val']
        self._set_path()
        self._set_data()

    def _set_path(self):
        if self.split == 'train':
            self.file_label_path = os.path.join(
                self.dataset_root, 'annotations', 'train.txt')
            self.imagefile_path = os.path.join(self.dataset_root,
                                               'ILSVRC2012_img_train')
        elif self.split == 'val':
            self.file_label_path = os.path.join(
                self.dataset_root, 'annotations', 'val.txt')
            self.imagefile_path = os.path.join(self.dataset_root,
                                               'ILSVRC2012_img_val')
        else:
            raise ValueError("unknown split mode")

    def _set_data(self):
        with open(self.file_label_path, 'r') as f:
            self.data = list(map(lambda x: x.split(), f.read().splitlines()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filepath, label = self.data[index]
        filepath = os.path.join(self.imagefile_path,
                                filepath)
        img = Image.open(filepath)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)
