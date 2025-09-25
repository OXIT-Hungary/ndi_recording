from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN


def hex_to_signed_int(hex_value):
    # int_value = int(hex_value, 16)
    if hex_value > 0x7FFF:  # Handle two’s complement negative values
        hex_value -= 0x10000
    return hex_value


def visca_to_euler(pan_int, tilt_int):
    pan_int = hex_to_signed_int(pan_int)
    tilt_int = hex_to_signed_int(tilt_int)

    pan_deg = pan_int / 16.0
    tilt_deg = tilt_int / 16.0

    return pan_deg, tilt_deg


def euler_to_visca(pan_deg, tilt_deg):

    pan_int = int(pan_deg * 16)
    tilt_int = int(tilt_deg * 16)

    return pan_int, tilt_int


def get_cluster_centroid(points: np.array, eps: float = 10.0, min_samples: int = 3):

    if len(points) < 2:
        return [], [], []

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Find largest cluster
    cluster_counts = defaultdict(int)
    for label in labels:
        if label != -1:  # Ignore noise
            cluster_counts[label] += 1

    if not cluster_counts:
        return [], [], []

    main_cluster = max(cluster_counts, key=cluster_counts.get)
    cluster_mask = labels == main_cluster
    cluster_points = points[cluster_mask]

    # Calculate gravity center
    cluster_center = cluster_points.mean(axis=0)

    return cluster_center, cluster_points, cluster_mask


def calc_pan_shift(bev_x_axis_line: int, x_axis_value: int, pan_distance: float) -> float:

    bev_percentage = (x_axis_value / bev_x_axis_line) * 100
    result_pan = pan_distance * (bev_percentage / 100)

    return result_pan


def save_homography(homography, img_name, path):
    split = img_name.split(".")
    name_without_sfx = split[0]
    homography_filename = f"{name_without_sfx}_homography.npy"
    h_path = os.path.join(path, homography_filename)
    np.save(h_path, homography)
