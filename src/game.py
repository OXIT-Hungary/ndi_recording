import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from enum import Enum
from filterpy.kalman import KalmanFilter

from src.utils.tmp import calc_pan_shift, euler_to_visca, visca_to_euler
from src.config import BEVConfig


class Goal:
    INTERNAL_HEIGHT = 0.9
    INTERNAL_WIDTH = 3.0
    NET_DEPTH = 1.1


class Track:
    class Status(Enum):
        TENTATIVE = 0
        CONFIRMED = 1
        DEAD = 2

    def __init__(self, t_id: int, x: float, y: float, dt: float) -> None:

        self.t_id = t_id
        self.kf = self._create_kalman_filter(x=x, y=y, dt=dt)

        self.confidence = 3
        self._status = Track.Status.TENTATIVE

    def predict(self) -> None:
        self.kf.predict()

        self.confidence = max(self.confidence - 1, 0)

    def update(self, z) -> None:
        self.kf.update(z)

        self.confidence = min(self.confidence + 2, 10)

        if self.confidence > 6 and self._status == Track.Status.TENTATIVE:
            self._status = Track.Status.CONFIRMED

    def get_direction(self):
        norm = np.linalg.norm([self.kf.x[2], self.kf.x[3]])
        if norm == 0:
            return np.array([0, 0])
        return np.array([self.kf.x[2], self.kf.x[3]]) / norm

    def _create_kalman_filter(self, x, y, dt=1.0):
        # Initialize Kalman Filter with 4 states (x, y, vx, vy) and 2 measurements (x, y)
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State Transition Matrix (F)
        kf.F = np.array(
            [
                [1, 0, dt, 0],  # x = x + vx*dt
                [0, 1, 0, dt],  # y = y + vy*dt
                [0, 0, 1, 0],  # vx = vx
                [0, 0, 0, 1],  # vy = vy
            ]
        )

        # Measurement Function (H) - we only measure position (x, y)
        kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )

        # Initial State Estimate: assume starting at origin with zero velocity
        kf.x = np.array([[x], [y], [0], [0]])

        # Initial Uncertainty in the State Estimate
        kf.P = np.diag([5.0, 5.0, 25.0, 25.0])

        # Measurement Noise Covariance Matrix (R)
        kf.R = np.diag([4.0, 4.0])

        # Process Noise Covariance Matrix (Q)
        q = 1.0  # process noise magnitude
        kf.Q = (
            np.array(
                [
                    [0.25 * dt**4, 0, 0.5 * dt**3, 0],
                    [0, 0.25 * dt**4, 0, 0.5 * dt**3],
                    [0.5 * dt**3, 0, dt**2, 0],
                    [0, 0.5 * dt**3, 0, dt**2],
                ]
            )
            * q
        )

        # No control input
        kf.B = None

        return kf

    @property
    def pos(self):
        return self.kf.x[:2]

    @property
    def speed(self):
        return self.kf.x[2:]

    def get_speed(self):
        return np.sqrt(self.kf.x[2] ** 2 + self.kf.x[3] ** 2)

    @property
    def status(self):
        return self._status


class Game:
    def __init__(self, config: BEVConfig):
        self.config = config

        self.court_width = self.config.court_size[0]
        self.court_height = self.config.court_size[1]

        self.H, _ = cv2.findHomography(
            config.points["image"], config.points["world"], method=0, confidence=0.99999, maxIters=100000
        )

        self._tracks = []
        self.t_id = 0

    def update(self, labels, boxes, scores) -> None:

        proj_points, labels, scores = self.project_to_bev(boxes, labels, scores)
        dets = proj_points[(labels == 2) & (scores > 0.4)].tolist()

        unmatched_det_inds = [i for i in range(len(dets))]

        # Propagate Tracks
        for track in self._tracks:
            track.predict()

        # Associate
        track_positions = np.array([t.pos.squeeze() for t in self._tracks])
        associations = self.associate(tracks=track_positions, dets=dets)

        # Update
        for t_ind, d_ind in associations:
            self._tracks[t_ind].update(z=dets[d_ind])
            unmatched_det_inds = unmatched_det_inds[:d_ind] + unmatched_det_inds[d_ind + 1 :]

        # Lifetime Management
        for ind, track in enumerate(self._tracks):
            if track.confidence == 0:
                self._tracks = self._tracks[:ind] + self._tracks[ind + 1 :]

        # Create new tracks
        for d_ind in unmatched_det_inds:
            self._tracks.append(Track(t_id=self.t_id, x=dets[d_ind][0], y=dets[d_ind][1], dt=1.0 / 5))
            self.t_id += 1

    def associate(self, tracks, dets, VI=None):
        if len(tracks) == 0 or len(dets) == 0:
            return []

        if VI is None:
            cost_matrix = cdist(tracks, dets, metric='euclidean')
        else:
            cost_matrix = cdist(tracks, dets, metric='mahalanobis', VI=VI)

        cost_matrix[cost_matrix > 2] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter assignments that were originally above the threshold
        valid_assignments = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= 1.5:
                valid_assignments.append((r, c))

        return valid_assignments

    def project_to_bev(self, boxes: np.array, labels: np.array, scores: np.array) -> np.array:

        if boxes.size == 0:
            return boxes, labels, scores

        proj_points = self._project_boxes(boxes)

        # mask out court size
        within_threshold = np.all(np.abs(proj_points) <= (self.config.court_size / 2), axis=1)
        return proj_points[within_threshold], labels[within_threshold], scores[within_threshold]

    def _project_boxes(self, boxes: np.array) -> np.array:
        """
        Project centroids onto the court.
        :param boxes: bounding box centroids. represented as [x, y, width, height]
        :param H: Homography matrix. From camera image coordinate systesm to BEV space
        """

        bbox_center = np.ones(shape=(len(boxes), 3))
        bbox_center[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        bbox_center[:, 1] = boxes[:, 3]

        projected_centers = np.matmul(self.H, bbox_center.T).T
        projected_centers /= projected_centers[:, 2][:, np.newaxis]

        return projected_centers[:, :2]

    def get_pan_from_bev(self, x_axis_value, presets):

        pan_left = presets['left'][0]  # configbol jonnek, pan left es right value of the presets
        pan_right = presets['right'][0]
        tilt = presets['left'][1]

        pan_deg_left, tilt_deg = visca_to_euler(pan_left, tilt)
        pan_deg_right, tilt_deg = visca_to_euler(pan_right, tilt)

        pan_deg_left = abs(pan_deg_left)
        pan_deg_right = abs(pan_deg_right)

        pan_distance = pan_deg_left + pan_deg_right

        res_pan = calc_pan_shift(self.court_width, x_axis_value, pan_distance)

        pan_int, tilt_int = euler_to_visca(res_pan, tilt_deg)

        pan_pos = pan_int + 65536 if pan_int < 3000 else pan_int
        tilt_pos = tilt_int + 65536 if tilt_int < 3000 else tilt_int

        return pan_pos, tilt_pos

    def calculate_reprojection_error(self):
        projected_pts = cv2.perspectiveTransform(self.config.image_points.reshape(-1, 1, 2), self.H).squeeze()

        return np.sqrt(np.sum((projected_pts - self.config.world_points) ** 2, axis=1)).mean()

    def coord_to_px(self, x: float, y: float, scale: int = 20) -> tuple[int, int]:
        """Convert court coords (meters) to image coords (pixels)."""
        x_px = int((x + self.court_width / 2 + self.config.court_padding[0]) * scale)
        y_px = int((self.court_height / 2 - y + self.config.court_padding[1]) * scale)

        return x_px, y_px

    def draw(self, detections: np.array = np.array([]), track_speeds=None, scale: int = 20):

        img = self.draw_court(scale=scale)
        img = self.draw_detections(img=img, dets=detections, track_speeds=track_speeds, scale=scale)

        return img

    def draw_court(self, scale: int = 20) -> np.ndarray:
        width_px = int((self.court_width + 2 * self.config.court_padding[0]) * scale)
        height_px = int((self.court_height + 2 * self.config.court_padding[1]) * scale)

        img = np.ones((height_px, width_px, 3), dtype=np.uint8) * np.array([229, 172, 2], dtype=np.uint8)

        half_w = self.court_width / 2
        half_h = self.court_height / 2

        # Draw half lines
        cv2.line(
            img,
            self.coord_to_px(x=0, y=-half_h, scale=scale),
            self.coord_to_px(x=0, y=half_h, scale=scale),
            (230, 230, 230),
            1,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w - self.config.court_padding[0], y=0, scale=scale),
            self.coord_to_px(x=half_w + self.config.court_padding[0], y=0, scale=scale),
            (230, 230, 230),
            1,
        )

        # Draw red boxes in corners
        cv2.rectangle(
            img,
            self.coord_to_px(x=-(half_w + 0.3 + 1.08), y=-half_h, scale=scale),
            self.coord_to_px(x=-(half_w + 0.3), y=-half_h + 2, scale=scale),
            (100, 100, 230),
            -1,
        )
        cv2.rectangle(
            img,
            self.coord_to_px(x=-(half_w + 0.3 + 1.08), y=half_h, scale=scale),
            self.coord_to_px(x=-(half_w + 0.3), y=half_h - 2, scale=scale),
            (100, 100, 230),
            -1,
        )
        cv2.rectangle(
            img,
            self.coord_to_px(x=(half_w + 0.3 + 1.08), y=-half_h, scale=scale),
            self.coord_to_px(x=(half_w + 0.3), y=-half_h + 2, scale=scale),
            (100, 100, 230),
            -1,
        )
        cv2.rectangle(
            img,
            self.coord_to_px(x=(half_w + 0.3 + 1.08), y=half_h, scale=scale),
            self.coord_to_px(x=(half_w + 0.3), y=half_h - 2, scale=scale),
            (100, 100, 230),
            -1,
        )

        # Draw top border line
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + self.config.court_padding[0]), y=half_h, scale=scale),
            self.coord_to_px(x=(half_w + self.config.court_padding[0]), y=half_h, scale=scale),
            (180, 180, 180),
            2,
        )

        cv2.line(
            img,
            self.coord_to_px(x=-half_w, y=half_h, scale=scale),
            self.coord_to_px(x=half_w, y=half_h, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w + 2, y=half_h, scale=scale),
            self.coord_to_px(x=half_w - 2, y=half_h, scale=scale),
            (0, 230, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w + 6, y=half_h, scale=scale),
            self.coord_to_px(x=half_w - 6, y=half_h, scale=scale),
            (0, 200, 0),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=half_w - 5.1, y=half_h, scale=scale),
            self.coord_to_px(x=half_w - 5, y=half_h, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w + 5.1, y=half_h, scale=scale),
            self.coord_to_px(x=-half_w + 5, y=half_h, scale=scale),
            (0, 0, 230),
            2,
        )

        # Draw bottom border line
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + self.config.court_padding[0]), y=-half_h, scale=scale),
            self.coord_to_px(x=(half_w + self.config.court_padding[0]), y=-half_h, scale=scale),
            (180, 180, 180),
            2,
        )

        cv2.line(
            img,
            self.coord_to_px(x=-half_w, y=-half_h, scale=scale),
            self.coord_to_px(x=half_w, y=-half_h, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w + 2, y=-half_h, scale=scale),
            self.coord_to_px(x=half_w - 2, y=-half_h, scale=scale),
            (0, 230, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w + 6, y=-half_h, scale=scale),
            self.coord_to_px(x=half_w - 6, y=-half_h, scale=scale),
            (0, 200, 0),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=half_w - 5.1, y=-half_h, scale=scale),
            self.coord_to_px(x=half_w - 5, y=-half_h, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w + 5.1, y=-half_h, scale=scale),
            self.coord_to_px(x=-half_w + 5, y=-half_h, scale=scale),
            (0, 0, 230),
            2,
        )

        # Draw Left Border
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + 0.3), y=half_h + self.config.court_padding[1], scale=scale),
            self.coord_to_px(x=-(half_w + 0.3), y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (180, 180, 180),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + 0.3), y=Goal.INTERNAL_WIDTH / 2 + 2, scale=scale),
            self.coord_to_px(x=-(half_w + 0.3), y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + 0.3), y=-half_h - self.config.court_padding[1], scale=scale),
            self.coord_to_px(x=-(half_w + 0.3), y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            (180, 180, 180),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + 0.3), y=-Goal.INTERNAL_WIDTH / 2 - 2, scale=scale),
            self.coord_to_px(x=-(half_w + 0.3), y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-(half_w + 0.3 + 1.08), y=-half_h - self.config.court_padding[1], scale=scale),
            self.coord_to_px(x=-(half_w + 0.3 + 1.08), y=half_h + self.config.court_padding[1], scale=scale),
            (180, 180, 180),
            2,
        )

        # Draw Left Goal
        cv2.line(
            img,
            self.coord_to_px(x=-half_w, y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            self.coord_to_px(x=-half_w - 1.3, y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (255, 255, 255),
            4,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w, y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            self.coord_to_px(x=-half_w - 1.3, y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            (255, 255, 255),
            4,
        )
        cv2.line(
            img,
            self.coord_to_px(x=-half_w - 1.3, y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            self.coord_to_px(x=-half_w - 1.3, y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (255, 255, 255),
            4,
        )

        # Draw Right Border
        cv2.line(
            img,
            self.coord_to_px(x=(half_w + 0.3), y=half_h + self.config.court_padding[1], scale=scale),
            self.coord_to_px(x=(half_w + 0.3), y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (180, 180, 180),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=(half_w + 0.3), y=Goal.INTERNAL_WIDTH / 2 + 2, scale=scale),
            self.coord_to_px(x=(half_w + 0.3), y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=(half_w + 0.3), y=-half_h - self.config.court_padding[1], scale=scale),
            self.coord_to_px(x=(half_w + 0.3), y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            (180, 180, 180),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=(half_w + 0.3), y=-Goal.INTERNAL_WIDTH / 2 - 2, scale=scale),
            self.coord_to_px(x=(half_w + 0.3), y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            (0, 0, 230),
            2,
        )
        cv2.line(
            img,
            self.coord_to_px(x=(half_w + 0.3 + 1.08), y=-half_h - self.config.court_padding[1], scale=scale),
            self.coord_to_px(x=(half_w + 0.3 + 1.08), y=half_h + self.config.court_padding[1], scale=scale),
            (180, 180, 180),
            2,
        )

        # Draw Right Goal
        cv2.line(
            img,
            self.coord_to_px(x=half_w, y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            self.coord_to_px(x=half_w + 1.3, y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (255, 255, 255),
            4,
        )
        cv2.line(
            img,
            self.coord_to_px(x=half_w, y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            self.coord_to_px(x=half_w + 1.3, y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            (255, 255, 255),
            4,
        )
        cv2.line(
            img,
            self.coord_to_px(x=half_w + 1.3, y=-Goal.INTERNAL_WIDTH / 2, scale=scale),
            self.coord_to_px(x=half_w + 1.3, y=Goal.INTERNAL_WIDTH / 2, scale=scale),
            (255, 255, 255),
            4,
        )

        return img

    def draw_detections(
        self,
        img: np.ndarray,
        dets: np.ndarray,
        track_speeds=None,
        scale: int = 20,
        cluster: bool = False,
        color=(0, 0, 0),
    ) -> np.ndarray:
        for det in dets:
            if not cluster:
                cv2.circle(
                    img,
                    center=self.coord_to_px(x=det[0], y=det[1], scale=scale),
                    radius=2,
                    color=color,
                    thickness=-1,
                )
            else:
                cv2.circle(
                    img,
                    center=self.coord_to_px(x=det[0], y=det[1], scale=scale),
                    radius=3,
                    color=(0, 255, 255),
                    thickness=1,
                )

        if track_speeds:
            for det, (direction, speed) in zip(dets, track_speeds):

                end_point = det + direction.squeeze() * speed

                cv2.line(
                    img,
                    pt1=self.coord_to_px(x=det[0], y=det[1], scale=scale),
                    pt2=self.coord_to_px(x=end_point[0], y=end_point[1], scale=scale),
                    color=(0, 255, 0) if speed > 0.35 else (0, 0, 255),
                    thickness=2,
                )

        return img

    @property
    def tracks(self) -> list:
        return [t for t in self._tracks if t.status == Track.Status.CONFIRMED]


def main(config):
    pass


if __name__ == "__main__":
    import argparse
    from src.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config path.", required=False, default="./default_config.yaml")

    args = parser.parse_args()
    cfg = load_config(file_path=args.config)

    main(config=cfg.bev)
