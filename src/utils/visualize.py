import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

class2color = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}
class2str = {1: 'Goalkeeper', 2: 'Player', 3: 'Referee'}


def draw(image, labels, boxes, scores, bucket_id, bucket_width, thrh=0.5):
    draw = ImageDraw.Draw(image)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    scr = scores
    lab = labels[scr > thrh]
    box = boxes[scr > thrh]

    left = bucket_id * bucket_width
    right = (bucket_id + 1) * bucket_width
    draw_overlay.rectangle([left, 0, right, image.height], fill=(0, 128, 255, 50))

    for box, label, score in zip(boxes, labels, scores):
        if score > thrh:
            draw.rectangle(box.tolist(), outline=class2color[label], width=2)
            draw.text((box[0], box[1]), text=class2str[label], fill="blue")

    blended = Image.alpha_composite(image.convert("RGBA"), overlay)
    cv2.imshow("Image", np.array(blended))
    cv2.waitKey(1)


class Visualization:

    def draw_waterpolo_court(self, args):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-args.court_width, args.court_width)
        ax.set_ylim(-args.court_height / 2, args.court_height / 2)
        ax.set_title("BEV")
        ax.grid(False)

        # Draw boundary with colored segments
        if args.draw_boundary:
            half_height = args.court_height / 2
            x_left = -args.court_width / 2
            x_right = args.court_width / 2

            half_width = args.court_width / 2
            y_bottom = -args.court_height / 2
            y_top = args.court_height / 2

            # Top and bottom boundaries
            ax.axhline(y=half_height, color='black', linewidth=2)
            ax.axhline(y=-half_height, color='black', linewidth=2)

            # Left boundary segments
            segments = [
                (-half_height, -half_height + 2, 'red'),
                (-half_height + 2, -half_height + 6.5, 'gray'),
                (-half_height + 6.5, -half_height + 8.5, 'red'),
                (-half_height + 8.5, half_height - 8.5, 'gray'),
                (half_height - 2, half_height - 6.5, 'gray'),
                (half_height - 6.5, half_height - 8.5, 'red'),
                (half_height - 8.5, -(half_height - 8.5), 'gray'),
                (half_height, half_height - 2, 'red'),
            ]
            for y_start, y_end, color in segments:
                ax.plot([x_left, x_left], [y_start, y_end], color=color, linewidth=5)

            # Right boundary segments
            segments = [
                (half_height, half_height - 2, 'red'),
                (half_height - 2, half_height - 6.5, 'gray'),
                (half_height - 6.5, half_height - 8.5, 'red'),
                (half_height - 8.5, -(half_height - 8.5), 'gray'),
                (-half_height + 2, -half_height + 6.5, 'gray'),
                (-half_height + 6.5, -half_height + 8.5, 'red'),
                (-half_height + 8.5, half_height - 8.5, 'gray'),
                (-half_height, -half_height + 2, 'red'),
            ]
            for y_start, y_end, color in segments:
                ax.plot([x_right, x_right], [y_start, y_end], color=color, linewidth=5)

            # bottom boundary segments
            segments = [
                (half_width, half_width - 2, 'red'),
                (half_width - 2, half_width - 6, 'yellow'),
                (-half_width + 6, half_width - 6, 'green'),
                (-half_width + 2, -half_width + 6, 'yellow'),
                (-half_width, -half_width + 2, 'red'),
            ]
            for x_start, x_end, color in segments:
                ax.plot([x_start, x_end], [y_bottom, y_bottom], color=color, linewidth=4)

            # # top boundary segments
            segments = [
                (half_width, half_width - 2, 'red'),
                (half_width - 2, half_width - 6, 'yellow'),
                (-half_width + 6, half_width - 6, 'green'),
                (-half_width + 2, -half_width + 6, 'yellow'),
                (-half_width, -half_width + 2, 'red'),
            ]
            for x_start, x_end, color in segments:
                ax.plot([x_start, x_end], [y_top, y_top], color=color, linewidth=4)

        # Draw half-distance line
        if args.draw_half_line:
            ax.axvline(0, color='black', linestyle='-', linewidth=1)

        # Draw 2-meter lines (red)
        if args.draw_2m_line:
            distance = 2.0
            x_left = -args.court_width / 2 + distance
            x_right = args.court_width / 2 - distance
            ax.axvline(x_left, color='red', linestyle='--', linewidth=1)
            ax.axvline(x_right, color='red', linestyle='--', linewidth=1)

        # Draw 5-meter lines (red)
        if args.draw_5m_line:
            distance = 5.0
            x_left = -args.court_width / 2 + distance
            x_right = args.court_width / 2 - distance
            ax.axvline(x_left, color='red', linestyle='--', linewidth=1)
            ax.axvline(x_right, color='red', linestyle='--', linewidth=1)

        # Draw 6-meter lines (green)
        if args.draw_6m_line:
            distance = 6.0
            x_left = -args.court_width / 2 + distance
            x_right = args.court_width / 2 - distance
            ax.axvline(x_left, color='green', linestyle='--', linewidth=1)
            ax.axvline(x_right, color='green', linestyle='--', linewidth=1)

        # points = get_four_points(fig, ax)
        # plt.show()
        return fig, ax

    def visualize_homography_result(self, img: npt.ArrayLike, homography: npt.ArrayLike) -> None:

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

    def draw_current_detection(self, players_in_bev, ax):
        if players_in_bev is not None:
            ax.scatter(players_in_bev[:, 0], players_in_bev[:, 1], color='red', marker='o', label='Detections')

    def draw_tracked_objects(self, active_tracks, ax):
        for track in active_tracks:
            x, y = track.kf.x[0], track.kf.x[1]
            ax.scatter(x, y, color='blue', marker='x')
            ax.text(x, y, f'ID: {track.track_id}', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
            # self.draw_velocity_vector(track, ax)

    def draw_velocity_vector(self, track, ax):
        x = track.kf.x[0]
        y = track.kf.x[1]
        vx = track.kf.x[2]
        vy = track.kf.x[3]
        ax.arrow(x, y, vx, vy, head_width=0.5, head_length=1, fc='green', ec='green')

    def draw_gravity_center(self, gravity_center, ax):
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

    def draw_centroid(self, centroid, ax, on_line=True):
        x = centroid[0]
        y = 0 if on_line else centroid[1]
        ax.scatter(x, y, color='red', marker='*', s=300, edgecolor='black', linewidth=1, label='Centroid')
