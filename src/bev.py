import cv2
import numpy as np
import itertools
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


from src.utils.tmp import calc_pan_shift, euler_to_visca, visca_to_euler
from src.config import BEVConfig


class Goal:
    INTERNAL_HEIGHT = 0.9
    INTERNAL_WIDTH = 3.0
    NET_DEPTH = 1.1


colors = {
    'green': [0.0, 0.4, 0.0],
    'red': [0.9, 0.0, 0.0],
    'yellow': [0.9, 0.9, 0.0],
    'black': [0.0, 0.0, 0.0],
    'white': [1.0, 1.0, 1.0],
}


class Field3D:
    def __init__(self, width: int = 25, height: int = 20) -> None:
        self.width = width
        self.height = height

        half_w = width / 2
        half_h = height / 2

        self.points = [[0, half_h, 0]]
        self.colors = [[1, 1, 1]]
        for i, j in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:

            new_points = [
                [i * (half_w - 6), j * half_h, 0],
                [i * (half_w - 5), j * half_h, 0],
                [i * (half_w - 2), j * half_h, 0],
                [i * (half_w + 0.3), j * half_h, 0],
                [i * (half_w + 0.3), j * (Goal.INTERNAL_WIDTH / 2 + 2), 0],
            ]

            new_colors = [
                colors["yellow"],
                colors['red'],
                colors['red'],
                colors['white'],
                colors['red'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
            ]

            if i * j > 0:
                goal_points = [
                    [i * (half_w + 0.3), j * (Goal.INTERNAL_WIDTH / 2), 0],
                    [i * (half_w + 0.3), j * (Goal.INTERNAL_WIDTH / 2), Goal.INTERNAL_HEIGHT],
                    [i * (half_w + 0.3 + Goal.NET_DEPTH), j * (Goal.INTERNAL_WIDTH / 2), 0],
                    [i * (half_w + 0.3 + Goal.NET_DEPTH), j * (Goal.INTERNAL_WIDTH / 2), Goal.INTERNAL_HEIGHT],
                ]

                new_points += goal_points
            else:
                goal_points = [
                    [i * (half_w + 0.3), j * (Goal.INTERNAL_WIDTH / 2), Goal.INTERNAL_HEIGHT],
                    [i * (half_w + 0.3), j * (Goal.INTERNAL_WIDTH / 2), 0],
                    [i * (half_w + 0.3 + Goal.NET_DEPTH), j * (Goal.INTERNAL_WIDTH / 2), Goal.INTERNAL_HEIGHT],
                    [i * (half_w + 0.3 + Goal.NET_DEPTH), j * (Goal.INTERNAL_WIDTH / 2), 0],
                ]

                new_points = (new_points + goal_points)[::-1]
                new_colors = new_colors[::-1]

            self.points += new_points
            self.colors += new_colors

        self.points[19:19] = [[0, -half_h, 0]]
        self.points = np.array(self.points)

        self.colors[19:19] = [colors['white']]
        self.colors = np.array(self.colors)

    def draw(self):
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Labels Example", 1024, 768)
        scene = gui.SceneWidget()
        window.add_child(scene)

        # Create scene and geometry
        scene.scene = rendering.Open3DScene(window.renderer)

        features, mat_features = self.create_features()
        lines, mat_lines = self.create_lines_side()
        plane, mat_plane = self.create_rectangle(center=(0, 0, -0.05), width=35, height=25)

        scene.scene.add_geometry("plane", plane, mat_plane)
        rec, mat_rec = self.create_rectangle(
            center=(self.width / 2 - 4, 0, 0), width=4, height=20, color=np.append(colors['yellow'], 0.5)
        )
        scene.scene.add_geometry("rec1", rec, mat_rec)
        rec, mat_rec = self.create_rectangle(
            center=(self.width / 2 - 1, 0, 0), width=2, height=20, color=np.append(colors['red'], 0.5)
        )
        scene.scene.add_geometry("rec2", rec, mat_rec)
        rec, mat_rec = self.create_rectangle(
            center=(-self.width / 2 + 4, 0, 0), width=4, height=20, color=np.append(colors['yellow'], 0.5)
        )
        scene.scene.add_geometry("rec3", rec, mat_rec)
        rec, mat_rec = self.create_rectangle(
            center=(-self.width / 2 + 1, 0, 0), width=2, height=20, color=np.append(colors['red'], 0.5)
        )
        scene.scene.add_geometry("rec4", rec, mat_rec)

        markings = o3d.geometry.LineSet()
        markings.points = o3d.utility.Vector3dVector(self.points)
        markings.lines = o3d.utility.Vector2iVector([[0, 19], [2, 17], [21, 36]])
        markings.colors = o3d.utility.Vector3dVector([colors['white'], colors['red'], colors['red']])

        mat_markings = rendering.MaterialRecord()
        mat_markings.shader = "unlitLine"
        mat_markings.line_width = 2.0

        scene.scene.add_geometry('markings', markings, mat_markings)
        scene.scene.add_geometry("lines", lines, mat_lines)
        scene.scene.add_geometry("points", features, mat_features)
        # scene.scene.add_geometry("line", line_set, material_line)

        # Add labels
        for i, pt in enumerate(self.points):
            scene.add_3d_label(pt, f"{i}")

        # Camera
        bounds = features.get_axis_aligned_bounding_box()
        scene.setup_camera(60, bounds, bounds.get_center())

        gui.Application.instance.run()

    def create_features(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 5.0

        return pcd, material

    def create_lines_side(self):

        lines = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [6, 8],
                [7, 9],
                [8, 9],
                [8, 10],
                [9, 11],
                [10, 11],
                [10, 12],
                [11, 13],
                [12, 13],
                [12, 14],
                [14, 15],
                [15, 16],
                [16, 17],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [21, 22],
                [22, 23],
                [23, 24],
                [24, 25],
                [25, 27],
                [25, 26],
                [26, 28],
                [27, 28],
                [28, 30],
                [27, 29],
                [29, 31],
                [29, 30],
                [30, 32],
                [31, 32],
                [31, 33],
                [33, 34],
                [34, 35],
                [35, 36],
                [36, 37],
                [37, 0],
                [7, 13],
                [26, 32],
            ]
        )

        clrs = np.array(
            [
                colors['green'],
                colors['yellow'],
                colors['yellow'],
                colors['red'],
                colors['white'],
                colors['red'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['red'],
                colors['white'],
                colors['red'],
                colors['yellow'],
                colors['yellow'],
                colors['green'],
                colors['green'],
                colors['yellow'],
                colors['yellow'],
                colors['red'],
                colors['white'],
                colors['red'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['black'],
                colors['red'],
                colors['white'],
                colors['red'],
                colors['yellow'],
                colors['yellow'],
                colors['green'],
                colors['black'],
                colors['black'],
            ]
        )

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(clrs)

        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 5.0

        return line_set, material

    def create_rectangle(self, center, width, height, color=[0, 0.67, 0.9, 0.9]):
        # Define 4 corners in the XY plane
        w, h = width / 2, height / 2
        corners = np.array(
            [
                [center[0] - w, center[1] - h, center[2]],
                [center[0] + w, center[1] - h, center[2]],
                [center[0] + w, center[1] + h, center[2]],
                [center[0] - w, center[1] + h, center[2]],
            ]
        )

        triangles = [[0, 1, 2], [0, 2, 3]]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(corners)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        # mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = color

        return mesh, material


class BEV:

    def __init__(self, config: BEVConfig):
        self.config = config

        self.court_width = self.config.court_size[0]
        self.court_height = self.config.court_size[1]

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

    def draw(self, detections: np.array = np.array([]), scale: int = 20):

        img = self.draw_court(scale=scale)
        img = self.draw_detections(img=img, dets=detections, scale=scale)

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

    def draw_detections(self, img: np.ndarray, dets: np.ndarray, scale: int = 20, cluster: bool = False) -> np.ndarray:
        for det in dets:
            if not cluster:
                cv2.circle(
                    img,
                    center=self.coord_to_px(x=det[0], y=det[1], scale=scale),
                    radius=2,
                    color=(0, 0, 0),
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

        return img


def main(config):

    field3d = Field3D()
    field3d.draw()


if __name__ == "__main__":
    import argparse
    from src.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config path.", required=False, default="./default_config.yaml")

    args = parser.parse_args()
    cfg = load_config(file_path=args.config)

    main(config=cfg.bev)
