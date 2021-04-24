import cv2
import numpy as np
from random import randint
from functools import reduce
from os import walk
from scipy.spatial import ConvexHull

DIMENSIONS = (512, 512)


def fragment_overlay(background_img, masked_fragment):
    mask = masked_fragment.astype(int).sum(-1) == np.zeros(DIMENSIONS)
    background_img = np.where(mask[..., None], background_img, masked_fragment)
    return background_img


def transparent_superimposition(background_img, masked_fragment):
    mask = masked_fragment.astype(int).sum(-1) == np.zeros(DIMENSIONS)
    background_img = np.where(mask[..., None], background_img, cv2.addWeighted(background_img, 0.5, masked_fragment, 0.5, 0))
    return background_img


def polygon_area(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def lab_adjust(image, delta_light=0, clip_limit=1.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    cl = clahe.apply(l)
    cl = cv2.add(cl, delta_light)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def rotate(image, angle=0, scale=1.0):
    center = tuple(ti//2 for ti in DIMENSIONS)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, DIMENSIONS)
    return rotated


def darken(image, delta_light=-25):
    return lab_adjust(image, delta_light=delta_light)


def lighten(image, delta_light=25):
    return lab_adjust(image, delta_light=delta_light)


def increase_contrast(image, clip_limit=2.5):
    return lab_adjust(image, clip_limit=clip_limit)


def blur(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return blurred


def do_nothing(image):
    return image


def random_image_adjustment():
    # change contrast, brightness, rotate?, blur? - functions with respective probabilities of being chosen
    chosen_transformations = []
    for list_of_functions in all_transformations:
        chosen_transformations.append(np.random.choice(list_of_functions, p=[0.6, 0.2, 0.2]))

    return compose_functions(*chosen_transformations)


def compose_functions(*func):
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, func, lambda x: x)


def load(path='images'):
    _, _, filenames = next(walk(path))
    return cv2.resize(np.array(list(map(lambda x: cv2.imread(path + '/' + x), filenames)), dtype='uint8'), DIMENSIONS)


def save(image, identifier=123):
    cv2.imwrite(f"results/{identifier}_{randint(0, 1000)}.jpg", image)


def random_point(shift=[255, 255], deviation=256):
    x, y = randint(shift[0] - deviation, shift[0] + deviation), randint(shift[1] - deviation, shift[1] + deviation)
    return np.array([x, y])


def random_mask():
    mask = np.zeros(DIMENSIONS, dtype='uint8')
    cv2.fillPoly(mask, pts=[random_polygon(9)], color=255)
    return mask


def random_polygon(n):
    points = np.random.randint(0, 511, size=(n, 2))
    hull = ConvexHull(points)
    return points[hull.vertices]


all_transformations = [[do_nothing, do_nothing, increase_contrast],
                       [do_nothing, darken, lighten],
                       [do_nothing, do_nothing, blur]]
