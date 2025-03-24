import numpy.typing as npt
import cv2
import imageio

class Dataset:
    
    def __init__(self):
        self.video_path = "dataset/input/cropped.mp4"
        self.model_path = "onnx/rtdetrv2.onnx"
        self.itk = 0
        self.filenames = []
    
    def read_a_test_image(
        self,
        img_path: str
        )-> npt.ArrayLike:
        """
        Read a test image.
        :param img_path: Path to the image.
        :return: Image.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found. Check the file path.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def save_result_img(self, plt):
        filename = f"dataset/output/frame_{self.itk}.png"
        plt.savefig(filename)  # Save current frame
        print(filename, " - finished")
        
        self.filenames.append(filename)
        
        self.itk += 1
        
    def create_and_save_gif(self):
        imageio.mimsave("animation.gif", [imageio.imread(f) for f in self.filenames], duration=0.1)