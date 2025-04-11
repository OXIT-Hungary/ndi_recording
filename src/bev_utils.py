import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Utils:

    def __init__(self):
        self.world_points = np.array([[-25, 10], [25, 10], [-25, -10], [25, -10]], dtype=np.float32)
        
        self.img_pts = np.array([[549, 103], [2315, 71], [73, 470], [2746, 407]], dtype=np.float32)
        #self.img_pts = np.array([[103, 549], [2315, 71], [73, 470], [2746, 407]], dtype=np.float32)
    def get_four_points(self, fig, ax):
        points = []

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'go', markersize=5)
                fig.canvas.draw()
                if len(points) >= 4:
                    plt.close()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return points

    def calculate_homography(self):

        img_pts = np.array(self.img_pts)
        world_pts = np.array(self.world_points)

        homography, _ = cv2.findHomography(img_pts, world_pts, method=0)

        return homography

    def calculate_reprojection_error(self, homography):

        img_pts = np.array(self.img_pts, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)

        projected_pts = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), homography)
        projected_pts = projected_pts.squeeze()

        error = np.sqrt(np.sum((projected_pts - world_pts) ** 2, axis=1)).mean()

        return error

    def save_homography(self, homography, img_name, path):
        split = img_name.split(".")
        name_without_sfx = split[0]
        homography_filename = f"{name_without_sfx}_homography.npy"
        h_path = os.path.join(path, homography_filename)
        np.save(h_path, homography)

        print(f"Homography saved to {homography_filename}")
