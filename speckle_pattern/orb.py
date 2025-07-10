import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def detect_orb(image, orb_parameters=None, mask=None, verbosityFlag=0):

    """
    Detect ORB features.

    Parameters
    ----------
    image : numpy.ndarray
        Image to detect the ORB features in.
    orb_parameters : dict(nfeatures = int, scaleFactor = float, nlevels = int, edgeTreshold = int, firstlevel = int, WTA_K = int, scoreType = int,  patchSize = int, fastTheshold = int) or None, default=None
        Parameters for the ORB method.
        - nfeatures: The number of best features to retain
        - scaleFactor: Pyramid decimation ratio, greater than 1
        - nlevels: 	the number of pyramid levels
        - edgeThreshold: the size of the border where the features are not detected
        - firstLevel: The level of pyramid to put source image to
        - WTA_K: The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we 
          take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4.
        - scoreType: scoreType	The default HARRIS_SCORE means that Harris algorithm is used to rank features
        - patchSize: size of the patch used by the oriented BRIEF descriptor
        - fastThreshold: the fast threshold
      
        If None: dict(nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstlevel = 0, WTA_K = 2, scoreType = 0,  patchSize = 31, fastTheshold = 20) is used.
    mask : numpy.ndarray
        Binary image defining the area (mask) to seacrh for corners.
    verbosityFlag : int, default=0
        Determine if the plots are shown or not.

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray of shape=(# points, 1, 2=(u, v)) of the detected feature points.        
    """
    # Parameters for ORB detection
    if orb_parameters is None:
        feature_params = dict(nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = 0,  patchSize = 31, fastThreshold = 20)
    else:
        feature_params = orb_parameters
        
    detector = cv.ORB_create(**feature_params)
    # kp = detector.detect(image, mask)
    # kp, des = detector.compute(image, kp)
    kp, des = detector.detectAndCompute(image, mask)

    points = np.zeros((1, 1, 2))
    for point_nr in range(len(kp)):
        points = np.append(points, np.array(kp[point_nr].pt).reshape((1, 1, 2)), axis=0)
    points = np.delete(points, 0, 0)

    if verbosityFlag >= 2:
        plot_orb(image, points, display=True)
    return points, kp, des

def plot_orb(image, points, display=False):
    nr_points = points.shape[0]
    vis = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    for u, v in np.float32(points).reshape(-1, 2):
       cv.circle(vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)
        
    if display:
        fig = plt.figure()
        plt.title(f'ORB feature points ({nr_points} points)')
        plt.imshow(vis)
        plt.show()

    return vis

if __name__ == "__main__":
    pass




    # """
    # Find feature points based on an ORB feature point detector.

    # :param img: input image.
    # :return: keypoints, descriptors.
    # """
    # # Initiate ORB detector
    # orb = cv.ORB_create()
    # # Find the keypoints and descriptors with ORB
    # kp, des = orb.detectAndCompute(img[0], None)
    # return kp, des
