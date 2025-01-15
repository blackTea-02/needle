from typing import List, Optional
from ..data_basic import Dataset
import struct
import gzip
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super(MNISTDataset, self).__init__(transforms)
        with gzip.open(image_filename, "rb") as image_file, gzip.open(label_filename, "rb") as label_file:
            magic_num_data, num_data, row, col = struct.unpack('>IIII', image_file.read(16))
            magic_num_label, num_label = struct.unpack('>II', label_file.read(8))
            images = np.frombuffer(image_file.read(), dtype='uint8').reshape((num_data, row * col)).astype('float32')
            range_ = np.max(images) - np.min(images)
            images = (images - np.min(images)) / range_
            labels = np.frombuffer(label_file.read(), dtype='uint8')
            self.images = images
            self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y = self.images[index], self.labels[index]
        if self.transforms:
            X = X.reshape((28, 28, -1))
            X = self.apply_transforms(X).reshape((-1, 28 * 28))
            return X, y
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION