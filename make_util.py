import cv2
import math
import numpy as np


def flip(img, mode):
    if mode == 'horizontal':
        t_img = cv2.flip(img, 1)
    elif mode == 'vertical':
        t_img = cv2.flip(img, 0)
    else:
        t_img = img
    return t_img


def logo(img, l_img, size, x, y):
    h, w, c = img.shape
    lh, lw, lc = l_img.shape

    r = math.sqrt((h * w * size) / (lh * lw * 100))
    nlh, nlw = int(r * lh), int(r * lw)

    lx = int((w - nlw) * x * 0.01)
    ly = int((h - nlh) * y * 0.01)

    l_img = cv2.resize(l_img, (nlw, nlh), interpolation=cv2.INTER_AREA)
    t_img = img

    roi = img[ly:ly + nlh, lx:lx + nlw]
    gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
    l_fg = cv2.bitwise_and(l_img, l_img, mask=mask_inv)

    result = cv2.bitwise_or(img_bg, l_fg)
    t_img[ly:ly + nlh, lx:lx + nlw] = result

    return t_img


def border(img, level):
    h, w, c = img.shape
    if level == 50:
        bh, bw = int(h / 4), int(w / 4)
        t_img = cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=0)
    elif level == 25:
        bh, bw = int(h / 8), int(w / 8)
        t_img = cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=0)
    else:
        t_img = img
    return t_img


def pip(img, p_img):
    h, w, c = img.shape
    ph, pw, pc = p_img.shape

    t_img = cv2.resize(p_img, (w, h), interpolation=cv2.INTER_AREA)
    resize_img = cv2.resize(img, (int(w * 2955 / pw), int(h * 1945 / ph)), interpolation=cv2.INTER_AREA)
    rh, rw, rc = resize_img.shape
    tx = int(w * 565 / pw)
    ty = int(h * 400 / ph)
    t_img[ty:ty + rh, tx:tx + rw] = resize_img

    return t_img


def rotate(img, degree):
    if degree == 90:
        t_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif degree == 180:
        t_img = cv2.rotate(img, cv2.ROTATE_180)
    elif degree == 270:
        t_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        t_img = img
    return t_img


def brightness(img, val):
    t_img = np.clip(img + float(val), 0, 255).astype(np.uint8)
    return t_img


def grayscale(img):
    t_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return t_img


def crop(img, val):
    h, w, c = img.shape
    my, mx = h // 2, w // 2
    nh, nw = round(h * val), round(w * val)
    ny, nx = my - nh // 2, mx - nw // 2
    t_img = img[ny: ny + nh, nx: nx + nw]
    return t_img
