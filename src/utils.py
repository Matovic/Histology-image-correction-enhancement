"""
A collection of Python functions and classes
"""


import cv2
import numpy as np
from typing import Any, Tuple, List
from matplotlib import pyplot as plt


__author__ = "Erik Matovic"
__version__ = "1.0"
__email__ = "xmatovice@stuba.sk"
__status__ = "Development"


def gamma_coorection(img: cv2.Mat, gamma:float) -> cv2.Mat:
    """
    Gamma correction
    """
    return pow(img, 1/gamma)


def equalize_hist(R: cv2.Mat, G: cv2.Mat, B: cv2.Mat) -> cv2.Mat:
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    
    # return merged image
    return cv2.merge((output1_R, output1_G, output1_B))


def plt_img(img1: cv2.Mat, img2: cv2.Mat, title1: str='H&M', title2: str='P63') -> None:
    """
    Plot for 2 images
    """
    plt.figure()
    f, axarr = plt.subplots(1,2)
    f.set_size_inches(18.5, 10.5)
    f.set_dpi(100)

    axarr[0].imshow(img1, cmap='gray')
    axarr[1].imshow(img2, cmap='gray')

    axarr[0].set_title(title1)
    axarr[1].set_title(title2)

    axarr[0].axis('off')
    axarr[1].axis('off')

    plt.show()


def calc_histogram_show(images: List, model: List) -> None:
    """
    :param images: List of images for histogram calculation
    :param model: List of used color model
    """
    plt.figure()
    f, axarr = plt.subplots(1,2)
    f.set_size_inches(18.5, 10.5)
    f.set_dpi(100)

    # indexes for subplots
    plt_index = 0

    # iterate through the list of images
    for img in images:
        # iterate through the colors of the BGR model
        for i, col in enumerate(model):
            # calculate a histogram of each color model for each image
            histr = cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0, 1], accumulate=False)
            # add to the subplot
            axarr[plt_index].plot(histr, color=col)
        # row iterate
        plt_index += 1
        # if its out of bound move a row
        if plt_index > 1:
            plt_index = 0

    axarr[0].set_title('H&M')
    axarr[1].set_title('P63')
    plt.show()

def show_img(img: cv2.Mat, txt: str) -> None:
    """
    Show images
    :param: img - image
    :param: txt - text of a window
    """
    cv2.imshow(txt, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_img(img: cv2.Mat, scale_percent: int) -> cv2.Mat:
    """
    Resizing images.
    :param: img - image
    :param: scale_percent - percent by which the image is resized
    :return: Resized image
    """
    # calculate the scale percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    img_resize = cv2.resize(img, dsize)
    return img_resize
