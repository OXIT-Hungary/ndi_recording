import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

class2color = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}
class2str = {1: 'Goalkeeper', 2: 'Player', 3: 'Referee'}

color_map = {'red': (0, 0, 255), 'gray': (128, 128, 128), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}


def draw_boxes(
    frame: np.ndarray, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, thrashold: float = 0.5
) -> np.ndarray:

    mask = scores > thrashold
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]

    for box, label, score in zip(boxes, labels, scores):
        cv2.rectangle(img=frame, pt1=(box[0], [1]), pt2=(box[2], [3]), color=class2color[label], thickness=2)

    return frame


def draw_bev():
    pass


def draw_waterpolo_court(court_width: float = 25.0, court_height: float = 20, padding: float = 5, scale: int = 20):
    def coord_to_px(x: float, y: float) -> tuple[int, int]:
        """Convert court coords (meters) to image coords (pixels)."""
        x_px = int((x + court_width / 2) * scale)
        y_px = int((court_height / 2 - y) * scale)
        return x_px, y_px

    width_px = int((court_width + padding) * scale)
    height_px = int((court_height) * scale)

    img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

    half_w = court_width / 2
    half_h = court_height / 2

    # Draw half lines
    cv2.line(img, coord_to_px(0, -half_h), coord_to_px(0, half_h), (150, 150, 150), 1)
    cv2.line(img, coord_to_px(-half_w, 0), coord_to_px(half_w, 0), (100, 100, 100), 1)

    # Draw top/bottom border lines
    cv2.line(img, coord_to_px(-12.5, half_h - 1), coord_to_px(12.5, half_h - 1), (0, 0, 230), 2)
    cv2.line(img, coord_to_px(-12.5, -half_h + 1), coord_to_px(12.5, -half_h + 1), (0, 0, 230), 2)

    cv2.line(img, coord_to_px(-10.5, half_h - 1), coord_to_px(10.5, half_h - 1), (0, 230, 230), 2)
    cv2.line(img, coord_to_px(-10.5, -half_h + 1), coord_to_px(10.5, -half_h + 1), (0, 230, 230), 2)

    cv2.line(img, coord_to_px(-7.5, half_h - 1), coord_to_px(-7.3, half_h - 1), (0, 0, 230), 2)
    cv2.line(img, coord_to_px(-7.5, -half_h + 1), coord_to_px(-7.3, -half_h + 1), (0, 0, 230), 2)
    cv2.line(img, coord_to_px(7.5, half_h - 1), coord_to_px(7.3, half_h - 1), (0, 0, 230), 2)
    cv2.line(img, coord_to_px(7.5, -half_h + 1), coord_to_px(7.3, -half_h + 1), (0, 0, 230), 2)

    cv2.line(img, coord_to_px(-half_w, half_h), coord_to_px(half_w, half_h), (0, 0, 0), 2)
    cv2.line(img, coord_to_px(-half_w, -half_h), coord_to_px(half_w, -half_h), (0, 0, 0), 2)
    cv2.line(img, coord_to_px(-6.5, half_h - 1), coord_to_px(7.5, half_h - 1), (0, 200, 0), 2)
    cv2.line(img, coord_to_px(-6.5, -half_h + 1), coord_to_px(7.5, -half_h + 1), (0, 200, 0), 2)

    # Draw left/right borders
    cv2.line(img, coord_to_px(-(half_w + 0.3), -half_h), coord_to_px(-(half_w + 0.3), half_h), (0, 0, 0), 2)
    cv2.line(img, coord_to_px(half_w + 0.3, -half_h), coord_to_px(half_w + 0.3, half_h), (0, 0, 0), 2)
    # cv2.line(img, coord_to_px(-half_w, half_h), coord_to_px(12.5, -half_h + 1), (0, 0, 230), 2)

    cv2.imshow('court', img)
    cv2.waitKey(0)


def visualize_homography_result(img: npt.ArrayLike, homography: npt.ArrayLike) -> None:

    h, w = img.shape[:2]

    # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    warped_image = cv2.warpPerspective(img, homography, (w, h))
    warped_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    # Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(warped_rgb)
    plt.title("Transformed Image")
    plt.axis("off")

    plt.show()


def draw_current_detection(players_in_bev, ax):
    if players_in_bev is not None:
        ax.scatter(players_in_bev[:, 0], players_in_bev[:, 1], color='red', marker='o', label='Detections')


def draw_tracked_objects(active_tracks, ax):
    for track in active_tracks:
        x, y = track.kf.x[0], track.kf.x[1]
        ax.scatter(x, y, color='blue', marker='x')
        ax.text(x, y, f'ID: {track.track_id}', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
        # self.draw_velocity_vector(track, ax)


def draw_velocity_vector(track, ax):
    x = track.kf.x[0]
    y = track.kf.x[1]
    vx = track.kf.x[2]
    vy = track.kf.x[3]
    ax.arrow(x, y, vx, vy, head_width=0.5, head_length=1, fc='green', ec='green')


def draw_gravity_center(gravity_center, ax):
    ax.scatter(
        gravity_center[0],
        gravity_center[1],
        color='gold',
        marker='*',
        s=300,
        edgecolor='black',
        linewidth=1,
        label='Gravity Center',
    )


def draw_centroid(centroid, ax, on_line=True):
    x = centroid[0]
    y = 0 if on_line else centroid[1]
    ax.scatter(x, y, color='red', marker='*', s=300, edgecolor='black', linewidth=1, label='Centroid')


if __name__ == "__main__":
    draw_waterpolo_court()
