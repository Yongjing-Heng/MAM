import numpy as np
import random
from PIL import Image

class RandomDisrupt(object):
    def __init__(self, cutting_num, p=0.5):
        self.cutting_num = cutting_num
        self.p = p

    def __call__(self, img):
        return disrupt_image(img, self.cutting_num, self.p)

def disrupt_image(img, cutting_num, p):
    imgnp = np.array(img)
    if img.mode == 'L':
        h, w = imgnp.shape
    elif img.mode == 'RGB' or img.mode == 'RGBA':
        h, w, c = imgnp.shape
    else:
        raise RuntimeError("Unrecognized image type. Image type should be 'L', 'RGB' or 'RGBA'.")

    a = random.random()
    """
    2 slices cutting
    """
    if cutting_num == 2 and a <= p:
        if a <= p / 2:
            up_half = imgnp[0:h // 2 + 1, 0:w]
            down_half = imgnp[h // 2 + 1:h, 0:w]
            imgnp = np.vstack((down_half, up_half))
        else:
            left_half = imgnp[0:h, 0:w // 2 + 1]
            right_half = imgnp[0:h, w // 2 + 1:w]
            imgnp = np.hstack((right_half, left_half))
    """
    3 slices cutting
    """
    if cutting_num == 3 and a <= p:
        if a <= p / 2:
            a_piece = imgnp[0:h // 3 + 1, 0:w]
            b_piece = imgnp[h // 3 + 1:2 * h // 3 + 1, 0:w]
            c_piece = imgnp[2 * h // 3 + 1:h, 0:w]
            if a <= p / 10:
                imgnp = np.vstack((c_piece, b_piece, a_piece))
            elif p / 10 < a <= 2 * p / 10:
                imgnp = np.vstack((c_piece, a_piece, b_piece))
            elif 2 * p / 10 < a <= 3 * p / 10:
                imgnp = np.vstack((a_piece, c_piece, b_piece))
            elif 3 * p / 10 < a <= 4 * p / 10:
                imgnp = np.vstack((b_piece, a_piece, c_piece))
            else:
                imgnp = np.vstack((b_piece, c_piece, a_piece))

        else:
            a_piece = imgnp[0:h, 0:w // 3 + 1]
            b_piece = imgnp[0:h, w // 3 + 1:2 * w // 3 + 1]
            c_piece = imgnp[0:h, 2 * w // 3 + 1:w]
            if p / 2 < a <= 6 * p / 10:
                imgnp = np.hstack((c_piece, b_piece, a_piece))
            elif 6 * p / 10 < a <= 7 * p / 10:
                imgnp = np.hstack((c_piece, a_piece, b_piece))
            elif 7 * p / 10 < a <= 8 * p / 10:
                imgnp = np.hstack((a_piece, c_piece, b_piece))
            elif 8 * p / 10 < a <= 9 * p / 10:
                imgnp = np.hstack((b_piece, a_piece, c_piece))
            else:
                imgnp = np.hstack((b_piece, c_piece, a_piece))
    """
    4 slices cutting
    """
    if cutting_num == 4 and a <= p:
#        if h % 2 == 0 and w % 2 == 0: imgnp = four_cutting_v1(imgnp, h, w, a, p)  # cross cutting
        imgnp = four_cutting_v2(imgnp, h, w, a, p)

    if cutting_num != 3 and cutting_num != 4 and cutting_num != 2:
        raise RuntimeError("The module 'RandomDisrupt' didn't work, please pick a right number for 'cutting_num'.")
    if img.mode == 'L':
        img = Image.fromarray(imgnp.astype('uint8')).convert('L')
        return img
    elif img.mode == 'RGB' or img.mode == 'RGBA':
        img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
        return img
    else:
        raise RuntimeError("Unrecognized image type. Image type should be 'L', 'RGB' or 'RGBA'.")
    

"""
cross cutting(fail)
"""
def four_cutting_v1(imgnp, h, w, a, p):
    a_piece = imgnp[0:h // 2, 0:w // 2]
    b_piece = imgnp[0:h // 2, w // 2:w]
    c_piece = imgnp[h // 2:h, 0:w // 2]
    d_piece = imgnp[h // 2:h, w // 2:w]

    if a <= p / 23:
        a_half = np.hstack((a_piece, b_piece))
        b_half = np.hstack((d_piece, c_piece))
        imgnp = np.vstack((a_half, b_half))
    elif p / 23 < a <= 2 * p / 23:
        a_half = np.hstack((a_piece, c_piece))
        b_half = np.hstack((b_piece, d_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 2 * p / 23 < a <= 3 * p / 23:
        a_half = np.hstack((a_piece, c_piece))
        b_half = np.hstack((d_piece, b_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 3 * p / 23 < a <= 4 * p / 23:
        a_half = np.hstack((a_piece, d_piece))
        b_half = np.hstack((b_piece, c_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 4 * p / 23 < a <= 5 * p / 23:
        a_half = np.hstack((a_piece, d_piece))
        b_half = np.hstack((c_piece, b_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 5 * p / 23 < a <= 6 * p / 23:
        a_half = np.hstack((b_piece, a_piece))
        b_half = np.hstack((c_piece, d_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 6 * p / 23 < a <= 7 * p / 23:
        a_half = np.hstack((b_piece, a_piece))
        b_half = np.hstack((d_piece, c_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 7 * p / 23 < a <= 8 * p / 23:
        a_half = np.hstack((b_piece, c_piece))
        b_half = np.hstack((a_piece, d_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 8 * p / 23 < a <= 9 * p / 23:
        a_half = np.hstack((b_piece, c_piece))
        b_half = np.hstack((d_piece, a_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 9 * p / 23 < a <= 10 * p / 23:
        a_half = np.hstack((b_piece, d_piece))
        b_half = np.hstack((a_piece, c_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 10 * p / 23 < a <= 11 * p / 23:
        a_half = np.hstack((b_piece, d_piece))
        b_half = np.hstack((c_piece, a_piece))
        imgnp = np.vstack((a_half, b_half))

    elif 11 * p / 23 < a <= 12 * p / 23:
        a_half = np.hstack((c_piece, a_piece))
        b_half = np.hstack((b_piece, d_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 12 * p / 23 < a <= 13 * p / 23:
        a_half = np.hstack((c_piece, a_piece))
        b_half = np.hstack((d_piece, b_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 13 * p / 23 < a <= 14 * p / 23:
        a_half = np.hstack((c_piece, b_piece))
        b_half = np.hstack((a_piece, d_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 14 * p / 23 < a <= 15 * p / 23:
        a_half = np.hstack((c_piece, b_piece))
        b_half = np.hstack((d_piece, a_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 15 * p / 23 < a <= 16 * p / 23:
        a_half = np.hstack((c_piece, d_piece))
        b_half = np.hstack((a_piece, b_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 16 * p / 23 < a <= 17 * p / 23:
        a_half = np.hstack((c_piece, d_piece))
        b_half = np.hstack((b_piece, a_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 17 * p / 23 < a <= 18 * p / 23:
        a_half = np.hstack((d_piece, a_piece))
        b_half = np.hstack((b_piece, c_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 18 * p / 23 < a <= 19 * p / 23:
        a_half = np.hstack((d_piece, a_piece))
        b_half = np.hstack((c_piece, b_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 19 * p / 23 < a <= 20 * p / 23:
        a_half = np.hstack((d_piece, b_piece))
        b_half = np.hstack((a_piece, c_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 20 * p / 23 < a <= 21 * p / 23:
        a_half = np.hstack((d_piece, b_piece))
        b_half = np.hstack((c_piece, a_piece))
        imgnp = np.vstack((a_half, b_half))
    elif 21 * p / 23 < a <= 22 * p / 23:
        a_half = np.hstack((d_piece, c_piece))
        b_half = np.hstack((a_piece, b_piece))
        imgnp = np.vstack((a_half, b_half))
    else:
        a_half = np.hstack((d_piece, c_piece))
        b_half = np.hstack((b_piece, a_piece))
        imgnp = np.vstack((a_half, b_half))

    return imgnp

"""
single direction cutting 
"""
def four_cutting_v2(imgnp, h, w, a, p):
    if a <= p / 2:
        a_piece = imgnp[0:h // 4 + 1, 0:w]
        b_piece = imgnp[h // 4 + 1:2 * h // 4 + 1, 0:w]
        c_piece = imgnp[2 * h // 4 + 1:3 * h // 4 + 1, 0:w]
        d_piece = imgnp[3 * h // 4 + 1:h, 0:w]
        if a <= p / 47:
            imgnp = np.vstack((a_piece, b_piece, d_piece, c_piece))
        elif p / 47 < a <= 2 * p / 47:
            imgnp = np.vstack((a_piece, c_piece, b_piece, d_piece))
        elif 2 * p / 47 < a <= 3 * p / 47:
            imgnp = np.vstack((a_piece, c_piece, d_piece, b_piece))
        elif 3 * p / 47 < a <= 4 * p / 47:
            imgnp = np.vstack((a_piece, d_piece, b_piece, c_piece))
        elif 4 * p / 47 < a <= 5 * p / 47:
            imgnp = np.vstack((a_piece, d_piece, c_piece, b_piece))
        elif 5 * p / 47 < a <= 6 * p / 47:
            imgnp = np.vstack((b_piece, a_piece, c_piece, d_piece))
        elif 6 * p / 47 < a <= 7 * p / 47:
            imgnp = np.vstack((b_piece, a_piece, d_piece, c_piece))
        elif 7 * p / 47 < a <= 8 * p / 47:
            imgnp = np.vstack((b_piece, c_piece, a_piece, d_piece))
        elif 8 * p / 47 < a <= 9 * p / 47:
            imgnp = np.vstack((b_piece, c_piece, d_piece, a_piece))
        elif 9 * p / 47 < a <= 10 * p / 47:
            imgnp = np.vstack((b_piece, d_piece, a_piece, c_piece))
        elif 10 * p / 47 < a <= 11 * p / 47:
            imgnp = np.vstack((b_piece, d_piece, c_piece, a_piece))
        elif 11 * p / 47 < a <= 12 * p / 47:
            imgnp = np.vstack((c_piece, a_piece, b_piece, d_piece))
        elif 12 * p / 47 < a <= 13 * p / 47:
            imgnp = np.vstack((c_piece, a_piece, d_piece, b_piece))
        elif 13 * p / 47 < a <= 14 * p / 47:
            imgnp = np.vstack((c_piece, b_piece, a_piece, d_piece))
        elif 14 * p / 47 < a <= 15 * p / 47:
            imgnp = np.vstack((c_piece, b_piece, d_piece, a_piece))
        elif 15 * p / 47 < a <= 16 * p / 47:
            imgnp = np.vstack((c_piece, d_piece, a_piece, b_piece))
        elif 16 * p / 47 < a <= 17 * p / 47:
            imgnp = np.vstack((c_piece, d_piece, b_piece, a_piece))
        elif 17 * p / 47 < a <= 18 * p / 47:
            imgnp = np.vstack((d_piece, a_piece, b_piece, c_piece))
        elif 18 * p / 47 < a <= 19 * p / 47:
            imgnp = np.vstack((d_piece, a_piece, c_piece, b_piece))
        elif 19 * p / 47 < a <= 20 * p / 47:
            imgnp = np.vstack((d_piece, b_piece, a_piece, c_piece))
        elif 20 * p / 47 < a <= 21 * p / 47:
            imgnp = np.vstack((d_piece, b_piece, c_piece, a_piece))
        elif 21 * p / 47 < a <= 22 * p / 47:
            imgnp = np.vstack((d_piece, c_piece, a_piece, b_piece))
        else:
            imgnp = np.vstack((d_piece, c_piece, b_piece, a_piece))

    if p / 2 < a:
        a_piece = imgnp[0:h, 0:w // 4 + 1]
        b_piece = imgnp[0:h, w // 4 + 1:2 * w // 4 + 1]
        c_piece = imgnp[0:h, 2 * w // 4 + 1:3 * w // 4 + 1]
        d_piece = imgnp[0:h, 3 * w // 4 + 1:w]
        if 23 * p / 47 < a <= 24 * p / 47:
            imgnp = np.hstack((a_piece, b_piece, c_piece, d_piece))
        elif 24 * p / 47 < a <= 25 * p / 47:
            imgnp = np.hstack((a_piece, b_piece, d_piece, c_piece))
        elif 25 * p / 47 < a <= 26 * p / 47:
            imgnp = np.hstack((a_piece, c_piece, b_piece, d_piece))
        elif 26 * p / 47 < a <= 27 * p / 47:
            imgnp = np.hstack((a_piece, c_piece, d_piece, b_piece))
        elif 27 * p / 47 < a <= 28 * p / 47:
            imgnp = np.hstack((a_piece, d_piece, b_piece, c_piece))
        elif 28 * p / 47 < a <= 29 * p / 47:
            imgnp = np.hstack((a_piece, d_piece, c_piece, b_piece))
        elif 29 * p / 47 < a <= 30 * p / 47:
            imgnp = np.hstack((b_piece, a_piece, c_piece, d_piece))
        elif 30 * p / 47 < a <= 31 * p / 47:
            imgnp = np.hstack((b_piece, a_piece, d_piece, c_piece))
        elif 31 * p / 47 < a <= 32 * p / 47:
            imgnp = np.hstack((b_piece, c_piece, a_piece, d_piece))
        elif 32 * p / 47 < a <= 33 * p / 47:
            imgnp = np.hstack((b_piece, c_piece, d_piece, a_piece))
        elif 33 * p / 47 < a <= 34 * p / 47:
            imgnp = np.hstack((b_piece, d_piece, a_piece, c_piece))
        elif 34 * p / 47 < a <= 35 * p / 47:
            imgnp = np.hstack((b_piece, d_piece, c_piece, a_piece))
        elif 35 * p / 47 < a <= 36 * p / 47:
            imgnp = np.hstack((c_piece, a_piece, b_piece, d_piece))
        elif 36 * p / 47 < a <= 37 * p / 47:
            imgnp = np.hstack((c_piece, a_piece, d_piece, b_piece))
        elif 37 * p / 47 < a <= 38 * p / 47:
            imgnp = np.hstack((c_piece, b_piece, a_piece, d_piece))
        elif 38 * p / 47 < a <= 39 * p / 47:
            imgnp = np.hstack((c_piece, b_piece, d_piece, a_piece))
        elif 39 * p / 47 < a <= 40 * p / 47:
            imgnp = np.hstack((c_piece, d_piece, a_piece, b_piece))
        elif 40 * p / 47 < a <= 41 * p / 47:
            imgnp = np.hstack((c_piece, d_piece, b_piece, a_piece))
        elif 41 * p / 47 < a <= 42 * p / 47:
            imgnp = np.hstack((d_piece, a_piece, b_piece, c_piece))
        elif 42 * p / 47 < a <= 43 * p / 47:
            imgnp = np.hstack((d_piece, a_piece, c_piece, b_piece))
        elif 43 * p / 47 < a <= 44 * p / 47:
            imgnp = np.hstack((d_piece, b_piece, a_piece, c_piece))
        elif 44 * p / 47 < a <= 45 * p / 47:
            imgnp = np.hstack((d_piece, b_piece, c_piece, a_piece))
        elif 45 * p / 47 < a <= 46 * p / 47:
            imgnp = np.hstack((d_piece, c_piece, a_piece, b_piece))
        else:
            imgnp = np.hstack((d_piece, c_piece, b_piece, a_piece))

    return imgnp
