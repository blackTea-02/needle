import numpy as np


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.fliplr(img)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        X对应第一个维度，Y对应第二个维度
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        ### BEGIN YOUR SOLUTION
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant',
                            constant_values=0)
        img_x, img_y = img.shape[0], img.shape[1]
        return padded_img[self.padding + shift_x:self.padding + shift_x + img_x,
               self.padding + shift_y:self.padding + shift_y + img_y, :]
        ### END YOUR SOLUTION
