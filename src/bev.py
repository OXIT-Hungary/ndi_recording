import cv2
import numpy as np
import onnxruntime

from src.utils.tmp import calc_pan_shift, euler_to_visca, visca_to_euler
from src.config import BEVConfig


class BEV:
    def __init__(self, config: BEVConfig):
        self.config = config

        self.H, _ = cv2.findHomography(config.points["image"], config.points["world"], method=0)

    def project_to_bev(self, boxes: np.array, labels: np.array, scores: np.array) -> np.array:

        proj_boxes = self._project_boxes(boxes)

        # mask out court size
        within_threshold = np.all(np.abs(proj_boxes) <= (self.config.court_size / 2), axis=1)

        # return proj_boxes, labels, scores
        return proj_boxes[within_threshold], labels[within_threshold], scores[within_threshold]

    def _project_boxes(self, boxes: np.array) -> np.array:
        """
        Project centroids onto the court.
        :param bboxes: bounding box centroids. represented as [x, y, width, height]
        :param H: Homography matrix. From camera image coordinate systesm to BEV space
        """

        bbox_center = np.ones(shape=(len(boxes), 3))
        bbox_center[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        bbox_center[:, 1] = boxes[:, 3]

        projected_centers = np.matmul(self.H, bbox_center.T).T
        projected_centers /= projected_centers[:, 2][:, np.newaxis]

        return projected_centers[:, :2]

    def get_pan_from_bev(self, x_axis_value, presets):

        bev_x_axis_line = 20

        pan_left_hexa = hex(presets['left'][0])  # configbol jonnek, pan left es right value of the presets
        pan_right_hexa = hex(presets['right'][0])
        tilt_hexa = hex(presets['left'][1])

        pan_deg_left, tilt_deg = visca_to_euler(pan_left_hexa, tilt_hexa)
        pan_deg_right, tilt_deg = visca_to_euler(pan_right_hexa, tilt_hexa)

        pan_deg_left = abs(pan_deg_left)
        pan_deg_right = abs(pan_deg_right)

        pan_distance = pan_deg_left + pan_deg_right
        res_pan = calc_pan_shift(bev_x_axis_line, x_axis_value, pan_distance)
        pan_hex, tilt_hex = euler_to_visca(res_pan, tilt_deg)

        return pan_hex, tilt_hex

    def calculate_reprojection_error(self):
        projected_pts = cv2.perspectiveTransform(self.config.image_points.reshape(-1, 1, 2), self.H).squeeze()

        return np.sqrt(np.sum((projected_pts - self.config.world_points) ** 2, axis=1)).mean()


def main(args):

    bev = BEV(args=args)

    homography = bev.utils.calculate_homography()
    rmse = bev.utils.calculate_reprojection_error(homography)
    print(rmse)

    video_capture = cv2.VideoCapture(bev.data.video_path)

    onnx_session = onnxruntime.InferenceSession(
        bev.data.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    filtered_positions = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        fig, ax = bev.vis.draw_waterpolo_court(bev.args)

        # ONNX - RT_DETR
        players_in_bev = bev.utils.onnx__inference(
            H=homography, frame_size=(2810, 730), img=img, onnx_session=onnx_session
        )

        # Update player tracking
        gravity_center, active_tracks = bev.tracker.update(players_in_bev if players_in_bev is not None else [])

        # gravity_center[0] += gravity_center[0] * 0.3

        # Draw centroid for camera movement
        if len(bev.centroid) != 0:
            bev.centroid = bev.utils.move_centroid_smoothly(bev.centroid, gravity_center)
            bev.vis.draw_centroid(bev.centroid, ax)
        else:
            bev.centroid = gravity_center

        # Visualization
        bev.vis.draw_current_detection(players_in_bev, ax)
        bev.vis.draw_tracked_objects(active_tracks, ax)
        bev.vis.draw_gravity_center(gravity_center, ax)

        bev.data.save_result_img(plt)

    bev.data.create_and_save_gif()

    print("GIF saved as animation.gif!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Draw a water polo court with enhanced markings.')
    parser.add_argument('--court-width', type=float, default=25.0, help='Width of the court in meters (default: 30)')
    parser.add_argument('--court-height', type=float, default=20.0, help='Height of the court in meters (default: 20)')
    parser.add_argument('--no-boundary', action='store_false', dest='draw_boundary', help='Disable court boundary')
    parser.add_argument(
        '--no-half-line', action='store_false', dest='draw_half_line', help='Disable half-distance line'
    )
    parser.add_argument('--no-2m', action='store_false', dest='draw_2m_line', help='Disable 2-meter lines')
    parser.add_argument('--no-5m', action='store_false', dest='draw_5m_line', help='Disable 5-meter lines')
    parser.add_argument('--no-6m', action='store_false', dest='draw_6m_line', help='Disable 6-meter lines')

    args = parser.parse_args()

    main(args)
