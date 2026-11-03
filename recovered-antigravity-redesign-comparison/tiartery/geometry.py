import cv2
import numpy as np

def resize_input(image, size, mode="stretch", mask=False):
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    h, w = image.shape
    if mode == "stretch":
        return cv2.resize(image, (size, size), interpolation=interpolation)
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    out = np.zeros((size, size), dtype=image.dtype)
    top, left = (size - nh)//2, (size - nw)//2
    out[top:top+nh, left:left+nw] = resized
    return out

def restore_probability(prob, shape, mode="stretch"):
    h, w = shape
    if mode == "letterbox":
        size = prob.shape[0]
        scale = min(size/h, size/w)
        nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
        top, left = (size-nh)//2, (size-nw)//2
        prob = prob[top:top+nh, left:left+nw]
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
