import numpy as np
import cv2
from albumentations import Compose, Normalize, Resize

def augment(aug, image):
    return aug(image=image)['image']

def load_image(img_data, pil=False):
    image = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print(img_data)
    return image

def make_square(img):
    if img.shape[0] > img.shape[1]:
        img = np.rollaxis(img, 1, 0)
    toppadlen = (img.shape[1] - img.shape[0])//2
    bottompadlen = img.shape[1] - img.shape[0] - toppadlen
    toppad = img[:5,:,:].mean(0, keepdims=True).astype(img.dtype)
    toppad = np.repeat(toppad, toppadlen, 0)
    bottompad = img[-5:,:,:].mean(0, keepdims=True).astype(img.dtype)
    bottompad = np.repeat(bottompad, bottompadlen, 0)
    return np.concatenate((toppad, img, bottompad), axis=0)

def pre_process(img_width, img_height):
    albumentations_transform = Compose([
        Resize(img_width, img_height), 
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
        ], p=1)
    return albumentations_transform

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
