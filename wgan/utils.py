import cv2
import numpy as np


def save_image(img, path, nrow=10):
    N, C, W, H = img.shape
    img2 = img.reshape([-1, W*nrow*nrow, H])
    img = img2[:, :W*nrow, :]
    for i in range(1, nrow):
        img = np.concatenate([img, img2[:, W*nrow*i:W*nrow*(i+1), :]], axis=2)
    img = (img + 1.0)/2.0 * 255
    img = img.transpose((1, 2, 0))
    cv2.imwrite(path, img)
