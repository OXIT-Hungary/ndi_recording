
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import cv2
import numpy.typing as npt
import os
import onnxruntime
import math
import scipy.interpolate

class Utils:
    
    def __init__(self):
        self.world_points = np.array([
            [-25, 10],
            [25, 10],
            [-25, -10],
            [25, -10]
        ], dtype=np.float32)

        self.img_pts = np.array([
            [549, 103],
            [2315, 71],
            [73, 470],
            [2746, 407]
        ], dtype=np.float32)
    
    def project_centroids_to_court(self, bboxes, labels, scores, H):
        """
        Project centroids onto the court.
        :param bboxes: bounding box centroids. represented as [x, y, width, height]
        :param H: Homography matrix. From camera image coordinate systesm to BEV space
        """
        projected_players_list = np.empty((1, 2), float)
        bboxes_player = bboxes[(labels == 2) & (scores > 0.5)]

        if len(bboxes_player) == 0:
            return None  # No players, default to middle band

        for bbox in bboxes:
            player_centroid = np.array([(bbox[0] + bbox[2])/2,bbox[3], 1]).reshape(3,1) 
            projected_centroids = np.matmul(H, player_centroid)
            projected_centroids_normalized = projected_centroids / projected_centroids[2]
            projected_players_list = np.append(projected_players_list, projected_centroids_normalized[0:2].T, axis=0)
        #print(projected_centroids_normalized)

        projected_players_list = np.delete(projected_players_list, 0, 0)
        return projected_players_list

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

        homography, _ = cv2.findHomography(img_pts, world_pts, method = 0)

        return homography

    def calculate_reprojection_error(self, homography):

        img_pts = np.array(self.img_pts, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)

        #r = np.array([1483,321]).reshape(1,2)
        #img_pts = np.vstack([img_pts, r])
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

    def onnx__inference(
        self,
        H: npt.ArrayLike = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        frame_size: npt.ArrayLike = (2810, 590),
        img: np.ndarray = None,
        frame: np.ndarray = None,
        onnx_session: onnxruntime.InferenceSession = None,
        ii: int = 0
    ) -> npt.ArrayLike:
        frame_size = np.array([[2810,  590]])

        labels, boxes, scores = onnx_session.run(
            output_names=None,
            input_feed={
                'images': img,
                "orig_target_sizes": frame_size,
            },
        )
        target_class_id = 2  
        #testing this shit
        bboxes_player = boxes[(labels == target_class_id) & (scores > 0.5)]

        # Draw the bounding boxes
        for box in bboxes_player:
            x_min, y_min, x_max, y_max = map(int, box)  # Convert to integers
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
            cv2.putText(
                frame,
                f"Class {target_class_id}: {scores[labels == target_class_id].max():.2f}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.imwrite("dataset/output/frame_" + str(ii) + "_f.jpg", frame)


        projected_players = self.project_centroids_to_court(boxes, labels, scores, H)

        return projected_players
    
    def lerp(self, t, times, points):
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        dt = (t-times[0]) / (times[1]-times[0])
        return dt*dx + points[0][0], dt*dy + points[0][1]
    
    def move_centroid_smoothly(self, current_pos, new_pos):
        
        current_pos = self.lerp(10, [1,100], [current_pos,new_pos])
        
        return current_pos