import numpy as np
import cv2
import matplotlib.pyplot as plt

# ToDo: add abstract class


class PointForFMatrix:

    @staticmethod
    def drawlines(img1, img2, lines, pts1, pts2):
        """img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines
        """

        r, c = img1.shape[:2]
        # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def cartesian_to_homogenous(self, arr):
        """ Convert catesian to homogenous points by appending a row of 1s
        :param arr: array of shape (num_dimension x num_points)
        :returns: array of shape ((num_dimension+1) x num_points)
        """
        if arr.ndim == 1:
            return np.hstack([arr, 1])
        return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

    def verify(self, matches, img1, img2, key_points_img1, key_points_img2, draw_points=False):
        pts1 = []  # pts - points
        pts2 = []

        for m in matches:
            pts1.append(key_points_img1[m.queryIdx].pt)
            pts2.append(key_points_img2[m.trainIdx].pt)

        pts1 = np.int32(pts1)  # ToDo: why?
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)  # cv2.FM_RANSAC = 8

        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]  # do flat mask and find where mask is 1
        pts2 = pts2[mask.ravel() == 1]

        # Find epilines corresponding to points in right image (second image)
        #
        lines_img1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines_img1 = lines_img1.reshape(-1, 3)
        img5, img6 = self.drawlines(img1, img2, lines_img1, pts1, pts2)

        # # Find epilines corresponding to points in left image (first image) and
        # # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self.drawlines(img2, img1, lines2, pts2, pts1)

        plt.subplot(121)
        plt.imshow(img5)
        plt.subplot(122)
        plt.imshow(img3)
        plt.show()

        # return self.cartesian_to_homogenous(pts1), self.cartesian_to_homogenous(pts2), F
        return pts1, pts2, F


