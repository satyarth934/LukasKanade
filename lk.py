import os
import cv2
import copy
import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(in_temp, in_temp_a, rectangle, s=np.zeros(2)):
    
    # x1, y1, x2, y2 = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
    x1, y1, x2, y2 = rectangle
    temp_y, temp_x = np.gradient(in_temp_a)

    # temp_x = cv2.Sobel(in_temp_a, cv2.CV_64F, 1, 0, ksize=3)
    # temp_y = cv2.Sobel(in_temp_a, cv2.CV_64F, 0, 1, ksize=3)

    # temp_x = cv2.convertScaleAbs(temp_x)
    # temp_y = cv2.convertScaleAbs(temp_y)


    ds = 1
    thresh = 0.001

    while np.square(ds).sum() > thresh:
        s_x, s_y = s[0], s[1]
        w_x1, w_y1, w_x2, w_y2 = x1 + s_x, y1 + s_y, x2 + s_x, y2 + s_y

        u = np.linspace(x1, x2, 87)
        v = np.linspace(y1, y2, 36)
        u0, v0 = np.meshgrid(u, v)

        w_u = np.linspace(w_x1, w_x2, 87)
        w_v = np.linspace(w_y1, w_y2, 36)
        w_u0, w_v0 = np.meshgrid(w_u, w_v)

        x = np.arange(0, in_temp.shape[0], 1)
        y = np.arange(0, in_temp.shape[1], 1)

        spline = RectBivariateSpline(x, y, in_temp)
        S = spline.ev(v0, u0)

        spline_a = RectBivariateSpline(x, y, in_temp_a)
        img_warp = spline_a.ev(w_v0, w_u0)


        error = S - img_warp
        img_error = error.reshape(-1, 1)


        spline_x = RectBivariateSpline(x, y, temp_x)
        warpTemp_x = spline_x.ev(w_v0, w_u0)

        spline_y = RectBivariateSpline(x, y, temp_y)
        warpTemp_y = spline_y.ev(w_v0, w_u0)
        temp = np.vstack((warpTemp_x.ravel(), warpTemp_y.ravel())).T


        jac_matrix = np.array([[1, 0], [0, 1]])


        hess_matrix = np.dot(temp, jac_matrix)

        H = np.dot(hess_matrix.T, hess_matrix)


        ds = np.dot(np.linalg.inv(H), np.dot((hess_matrix.T), img_error))


        s[0] += ds[0, 0]
        s[1] += ds[1, 0]

    stop = s
    return stop
