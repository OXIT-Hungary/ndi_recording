import cv2
import numpy as np
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
