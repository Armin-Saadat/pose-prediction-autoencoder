import numpy as np
import matplotlib
import cv2


def draw_keypoints_posetrack(outputs):
    edges = [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (4, 6), (5, 7), (2, 8), (3, 9), (8, 9), (8, 10), (9, 11), (10, 12),
             (11, 13)]
    image = np.zeros((1080, 1920, 3))
    keypoints = outputs
    keypoints = keypoints.reshape(-1, 2)
    for p in range(keypoints.shape[0]):
        if not (keypoints[p, 0] <= 0 or keypoints[p, 1] <= 0):
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
                       lineType=cv2.FILLED)
            cv2.putText(image, f"{p}", (int(keypoints[p, 0] + 10), int(keypoints[p, 1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
    for ie, e in enumerate(edges):
        rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
        rgb = rgb * 255
        # join the keypoint pairs to draw the skeletal structure
        if not ((keypoints[e, 0][0] <= 0 or keypoints[e, 1][0] <= 0) or (
                keypoints[e, 0][1] <= 0 or keypoints[e, 1][1] <= 0)):
            cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]), (keypoints[e, 0][1], keypoints[e, 1][1]),
                     tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image


def draw_local_keypoint(outputs):
    edges = [(1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (8, 10), (9, 11),
             (10, 12)]

    image = np.zeros((1080, 1920, 3))
    keypoints = outputs
    keypoints = keypoints.reshape(-1, 2)
    for p in range(keypoints.shape[0]):
        cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
                   lineType=cv2.FILLED)
    for ie, e in enumerate(edges):
        rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
        rgb = rgb * 255
        # join the keypoint pairs to draw the skeletal structure
        if not ((keypoints[e, 0][0] <= 0 or keypoints[e, 1][0] <= 0) or (
                keypoints[e, 0][1] <= 0 or keypoints[e, 1][1] <= 0)):
            cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]), (keypoints[e, 0][1], keypoints[e, 1][1]),
                     tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image


def draw_keypoints(outputs):
    edges = [(13, 15), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (8, 10),
             (7, 9), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    image = np.zeros((1080, 1920, 3))
    for i in range(len(outputs)):
        keypoints = outputs[i]
        keypoints = keypoints.reshape(-1, 2)
        for p in range(keypoints.shape[0]):
            if not (keypoints[p, 0] == 0 and keypoints[p, 1] == 0):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
                           lineType=cv2.FILLED)
        for ie, e in enumerate(edges):
            rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
            rgb = rgb * 255
            if not ((keypoints[e, 0][0] == 0 and keypoints[e, 1][0] == 0) or (
                    keypoints[e, 0][1] == 0 and keypoints[e, 1][1] == 0)):
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])), tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image


def draw_keypoints_op(outputs):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (8, 12), (9, 10), (10, 11),
             (12, 13), (13, 14), (11, 24), (11, 22), (22, 23), (14, 21), (14, 19), (19, 20), (0, 15), (0, 16), (15, 17),
             (16, 18)]
    image = np.zeros((1080, 1920, 3))
    keypoints = outputs
    keypoints = keypoints.reshape(-1, 2)
    for p in range(keypoints.shape[0]):
        if not (keypoints[p, 0] <= 0 or keypoints[p, 1] <= 0):
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
                       lineType=cv2.FILLED)
    for ie, e in enumerate(edges):
        rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
        rgb = rgb * 255
        if not ((keypoints[e, 0][0] <= 0 and keypoints[e, 1][0] <= 0) or (
                keypoints[e, 0][1] <= 0 and keypoints[e, 1][1] <= 0)):
            cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]), (keypoints[e, 0][1], keypoints[e, 1][1]),
                     tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image
