import cv2
import numpy as np

from src.utils.tmp import calc_pan_shift, euler_to_visca, visca_to_euler
from src.config import BEVConfig


class Goal:
    INTERNAL_HEIGHT = 0.9
    INTERNAL_WIDTH = 3.0
    NET_DEPTH = 1.1


class BEV:

    def __init__(self, config: BEVConfig):
        self.config = config

        self.court_width = self.config.court_size[0]
        self.court_height = self.config.court_size[1]

        self.H, _ = cv2.findHomography(
            config.points["image"], config.points["world"], method=0, confidence=0.99999, maxIters=100000
        )

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
                    color=(0, 255, 0) if speed > 0.5 else (0, 0, 255),
                    thickness=2,
                )

        return img


def main(config):

    bev = BEV(config=config)
    bev.draw()


if __name__ == "__main__":
    import argparse
    from src.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config path.", required=False, default="./default_config.yaml")

    args = parser.parse_args()
    cfg = load_config(file_path=args.config)

    main(config=cfg.bev)
