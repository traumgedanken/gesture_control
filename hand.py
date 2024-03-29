import math

import cv2
import numpy as np


class Hand:
    def __init__(self, binary, masked, raw, frame):
        self.masked = masked
        self.binary = binary
        self._raw = raw
        self.frame = frame
        self.contours = []
        self.outline = self.draw_outline()
        self.outline[-150:, :150, 1] = self.binary

    def draw_outline(self, min_area=10000, color=(0, 255, 0), thickness=2):
        contours, _ = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        palm_area = 0
        flag = None
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > palm_area:
                palm_area = area
                flag = i
        if flag is not None and palm_area > min_area:
            cnt = contours[flag]
            self.contours = cnt
            cpy = self.frame.copy()
            cv2.drawContours(cpy, [cnt], 0, color, thickness)
            self.binary = self.crop_contour(self.binary, cnt)
            return cpy
        else:
            return self.frame

    def crop_contour(self, orig, cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        size = max(w, h)
        y_start = int(y - size / 2 + h / 2)
        x_start = int(x - size / 2 + w / 2)
        cropped = orig[
            max(0, y_start) : min(orig.shape[1] - 1, y_start + size),
            max(0, x_start) : max(orig.shape[0] - 1, x_start + size),
        ]
        return cv2.resize(cropped, (150, 150))

    def extract_fingertips(self, filter_value=50):
        cnt = self.contours
        if len(cnt) == 0:
            return cnt
        points = []
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            end = tuple(cnt[e][0])
            points.append(end)
        filtered = self.filter_points(points, filter_value)

        filtered.sort(key=lambda point: point[1])
        return [pt for idx, pt in zip(range(5), filtered)]

    def filter_points(self, points, filter_value):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i] and points[j] and self.dist(points[i], points[j]) < filter_value:
                    points[j] = None
        filtered = []
        for point in points:
            if point is not None:
                filtered.append(point)
        return filtered

    def get_center_of_mass(self):
        if len(self.contours) == 0:
            return None
        M = cv2.moments(self.contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    @staticmethod
    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (b[1] - a[1]) ** 2)
