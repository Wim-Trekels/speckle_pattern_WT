'''
Functions for KAZE detection.
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def detect_kaze(image, kaze_parameters=None, mask=None):
    """
    Detect KAZE features.

    Parameters
    ----------
    image : numpy.ndarray
        Image to detect the KAZE features in.
    kaze_parameters : dict(extended=bool, upright=bool, threshold=float, nOctaves=int, nOctaveLayers=int, diffusivity=int) or None, default=None
        Parameters for the KAZE method.
        - extended: set to enable extraction of extended (128-byte) descriptor.
        - upright: set to enable use of upright descriptors (non rotation-invariant).
        - threshold: detector response threshold to accept point
        - nOctaves: maximum octave evolution of the image
        - nOctaveLayers: default number of sublevels per scale level
        - diffusivity: diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        If None: dict(extended=False, upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=1) is used.
    mask : numpy.ndarray
        Binary image defining the area (mask) to seacrh for corners.


    Returns
    -------
    numpy.ndarray
        A numpy.ndarray of shape=(# points, 1, 2=(u, v)) of the detected feature points.        
    """
    # Parameters for KAZE detection
    if kaze_parameters is None:
        feature_params = dict(extended=False, upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=1)
    else:
        feature_params = kaze_parameters
        
    detector = cv.KAZE.create(**feature_params)
    # kp = detector.detect(image, mask)
    # kp, des = detector.compute(image, kp)
    kp, des = detector.detectAndCompute(image, mask)

    points = np.zeros((1, 1, 2))
    for point_nr in range(len(kp)):
        points = np.append(points, np.array(kp[point_nr].pt).reshape((1, 1, 2)), axis=0)
    points = np.delete(points, 0, 0)

    return points, kp, des

def plot_kaze(image, points, display=False):
    nr_points = points.shape[0]
    vis = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    for u, v in np.float32(points).reshape(-1, 2):
       cv.circle(vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)
        
    if display:
        fig = plt.figure()
        plt.title(f'KAZE feature points ({nr_points} points)')
        plt.imshow(vis)
        plt.show()

    return vis

if __name__ == "__main__":
    pass

