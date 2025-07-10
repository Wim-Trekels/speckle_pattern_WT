'''
Functions for SIFT detection.
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def detect_sift(image, sift_parameters=None, mask=None, verbosityFlag=0):
    """
    Detect SIFT features.

    Parameters
    ----------
    image : numpy.ndarray
        Image to detect the SIFT features in.
    sift_parameters : dict(nfeatures=int, nOctaveLayers = int, contactThreshold=float, edgeTreshold = float, sigma = float, descriptorType) or None, default=None
        Parameters for the SIFT method.
        - nfeatures: The number of best features to retain
        - nOctaveLayers: default number of sublevels per scale level
        - contrastTreshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        - edgeTreshold: The threshold used to filter out edge-like features
        - sigma: The sigma of the Gaussian applied to the input image at the octave #0
        - descriptorType: The type of descriptors
  
        If None: dict(nfeatures=100, nOctaveLayers=4, contrastThreshold=1, edgeThreshold=10, sigma=1.6, descriptorType=cv.CV_32F) is used.
    mask : numpy.ndarray
        Binary image defining the area (mask) to seacrh for corners.
    verbosityFlag : int, default=0
        Determine if the plots are shown or not.

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray of shape=(# points, 1, 2=(u, v)) of the detected feature points.        
    """
    # Parameters for SIFT detection
    if sift_parameters is None:
        feature_params = dict(nfeatures=100, nOctaveLayers=4, contrastThreshold=1, edgeThreshold=10, sigma=1.6, descriptorType=cv.CV_32F)
    else:
        feature_params = sift_parameters
        
    detector = cv.SIFT_create(**feature_params)
    # kp = detector.detect(image, mask)
    # kp, des = detector.compute(image, kp)
    kp, des = detector.detectAndCompute(image, mask)

    points = np.zeros((1, 1, 2))
    for point_nr in range(len(kp)):
        points = np.append(points, np.array(kp[point_nr].pt).reshape((1, 1, 2)), axis=0)
    points = np.delete(points, 0, 0)

    if verbosityFlag >= 2:
        plot_sift(image, points, display=True)
    return points, kp, des

def plot_sift(image, points, display=False):
    nr_points = points.shape[0]
    vis = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    for u, v in np.float32(points).reshape(-1, 2):
       cv.circle(vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)
        
    if display:
        fig = plt.figure()
        plt.title(f'SIFT feature points ({nr_points} points)')
        plt.imshow(vis)
        plt.show()

    return vis

if __name__ == "__main__":
    pass

