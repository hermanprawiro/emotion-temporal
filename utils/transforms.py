import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def get_color_distortion(s=1.0):
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])

    return color_distort

def get_gaussian_blur(img):
    img = np.array(img)
    img_blur = cv2.GaussianBlur(img, (15, 15), 0)
    return Image.fromarray(img_blur)