from typing import Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from utils import ObjectInfo, PointCloud, extract_largest_contour


SIZE_SOUP = np.array([0.07, 0.07, 0.1])
DEBUG = False


def segment_soup(
    img_color: np.ndarray,
    img_depth: np.ndarray,
    point_cloud: PointCloud,
    mask_others: np.ndarray,
) -> Tuple[np.ndarray, Optional[ObjectInfo]]:
    # Filter color range.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    mask_metal = (40 < img_h) & (img_h < 100)
    mask_metal &= (5 < img_s) & (img_s < 40)
    mask_metal &= (100 < img_v) & (img_v < 170)

    mask_red = (170 < img_h) | (img_h < 25)
    mask_red &= (100 < img_s) & (img_s < 250)
    mask_red &= (70 < img_v) & (img_v < 160)

    # mask_soup = (mask_metal | mask_red) & ~mask_others
    mask_soup =  mask_red & ~mask_others

    if DEBUG:
        mask_soup_color = mask_soup

    mask_soup = extract_largest_contour(mask_soup)

    ds = img_depth[mask_soup]
    if len(ds) == 0:
        return mask_soup, None
    d_min = ds.min() - 20
    d_max = ds.max() + 20
    mask_depth = (d_min <= img_depth) & (img_depth <= d_max)

    idx_points = point_cloud.flatten_indices(mask_soup)
    xyzs = point_cloud.points[idx_points]
    xyz_min = xyzs.min(axis=0)
    xyz_min[2] = 0
    xyz_max = xyzs.max(axis=0) + np.array([0.01, 0, 0.02])
    idxx_min = (xyz_min[None, :] <= point_cloud.points).all(axis=1)
    idxx_max = (point_cloud.points <= xyz_max[None, :]).all(axis=1)
    mask_xyz = point_cloud.expand_indices(idxx_min & idxx_max)

    mask_soup = mask_depth & mask_xyz & ~mask_others

    mask_soup = extract_largest_contour(mask_soup)

    # Get object pose.
    pos = xyzs.mean(axis=0)
    pos[0] = xyz_max[0] - SIZE_SOUP[0] / 2
    zs = xyzs[:, 0]
    pos[1] = np.median(xyzs[zs > np.percentile(zs, 99), 1])
    pos[2] = max(0, xyz_min[2]) + SIZE_SOUP[2] / 2

    if DEBUG:
        cv2.imshow("Color", img_color)
        cv2.imshow("h", img_h)
        cv2.imshow("s", img_s)
        cv2.imshow("v", img_v)
        cv2.imshow("red", mask_red.astype(np.uint8) * 255)
        cv2.imshow("metal", mask_metal.astype(np.uint8) * 255)
        cv2.imshow("color", mask_soup_color.astype(np.uint8) * 255)
        cv2.imshow("others", mask_others.astype(np.uint8) * 255)
        cv2.imshow("depth", mask_depth.astype(np.uint8) * 255)
        cv2.imshow("xyz", mask_xyz.astype(np.uint8) * 255)
        cv2.imshow("soup", mask_soup.astype(np.uint8) * 255)
        key = cv2.waitKey(0)
        if key is not None and chr(key) == "q":
            exit(1)

    return mask_soup, ObjectInfo(pos=pos, size=SIZE_SOUP)
