import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import onnxruntime

from src.bev.visualization import Visualization
from src.bev.utils import Utils
from src.bev.player_tracker import Track
from src.bev.player_tracker import Tracker
from src.bev.data import Dataset

import time

def get_args():
    parser = argparse.ArgumentParser(description='Draw a water polo court with enhanced markings.')
    parser.add_argument('--court-width', type=float, default=25.0, 
                    help='Width of the court in meters (default: 30)')
    parser.add_argument('--court-height', type=float, default=20.0, 
                    help='Height of the court in meters (default: 20)')
    parser.add_argument('--no-boundary', action='store_false', dest='draw_boundary', 
                    help='Disable court boundary')
    parser.add_argument('--no-half-line', action='store_false', dest='draw_half_line', 
                    help='Disable half-distance line')
    parser.add_argument('--no-2m', action='store_false', dest='draw_2m_line', 
                    help='Disable 2-meter lines')
    parser.add_argument('--no-5m', action='store_false', dest='draw_5m_line', 
                    help='Disable 5-meter lines')
    parser.add_argument('--no-6m', action='store_false', dest='draw_6m_line', 
                    help='Disable 6-meter lines')
    
    args = parser.parse_args()
    
    return args

class BEV():
    
    def __init__(self, args):
        self.vis = Visualization()
        self.utils = Utils()
        #self.track = Track()
        self.tracker = Tracker(max_age=25, min_hits=3)
        self.data = Dataset()
        self.args = args
        self.centroid = []
        self.start_time = time.time()
        self.end_time = 0
        
        self.homography = self.utils.calculate_homography()
        #rmse = self.utils.calculate_reprojection_error(homography)
        #self.onnx_session = onnxruntime.InferenceSession(self.data.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) 
      
    def process_frame(self, frame, onnx_session, ii, debug = False):

        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        
        if debug: fig, ax = self.vis.draw_waterpolo_court(self.args)
        
        # ONNX - RT_DETR
        players_in_bev = self.utils.onnx__inference(
            H = self.homography,
            frame_size = (2810, 590),
            img = img,
            frame = frame,
            onnx_session = onnx_session,
            ii = ii
        )
        
        #outlier filtering
        idx_list = []
        _dist_threshold = 12.5
        if players_in_bev is not None:
            for k in range(len(players_in_bev)):
                if players_in_bev[k][0] < -_dist_threshold or players_in_bev[k][0] > _dist_threshold or players_in_bev[k][1] < -_dist_threshold or players_in_bev[k][1] > _dist_threshold:
                    idx_list.append(k)
            players_in_bev = np.delete(players_in_bev, idx_list, axis=0)

            # Update player tracking
            gravity_center, active_tracks = self.tracker.update(players_in_bev if players_in_bev is not None else [])
                    
            #if players_in_bev != None :
                
            # Draw centroid for camera movement
            if not None in gravity_center:
                if len(self.centroid) != 0:
                    self.centroid = self.utils.move_centroid_smoothly(self.centroid, gravity_center)
                    if debug: self.vis.draw_centroid(self.centroid, ax)
                else:
                    self.centroid = gravity_center

            print(self.centroid)
        
            # Debug - Time
            """ self.end_time = time.time()
            print("Full cycle in sec: ",self.end_time - self.start_time)
            self.start_time = time.time() """
            
            # Visualization
            if debug:
                self.vis.draw_current_detection(players_in_bev, ax)
                self.vis.draw_tracked_objects(active_tracks, ax)
                self.vis.draw_gravity_center(gravity_center, ax)
                self.data.save_result_img(plt)
                plt.cla()
                plt.close()

        return self.centroid
        
      
    def bev_main(self):

        homography = self.utils.calculate_homography()
        rmse = self.utils.calculate_reprojection_error(homography)
        print(rmse)

        video_capture = cv2.VideoCapture(self.data.video_path)

        onnx_session = onnxruntime.InferenceSession(self.data.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) 
        
        filtered_positions = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            
            fig, ax = self.vis.draw_waterpolo_court(self.args)

            # ONNX - RT_DETR
            players_in_bev = self.utils.onnx__inference(
                H = homography,
                frame_size = (2810, 730),
                img = img,
                onnx_session = onnx_session
            )
            
            # Update player tracking
            gravity_center, active_tracks = self.tracker.update(players_in_bev if players_in_bev is not None else [])
                
            # Draw centroid for camera movement
            if len(self.centroid) != 0 :
                self.centroid = self.utils.move_centroid_smoothly(self.centroid, gravity_center)
                self.vis.draw_centroid(self.centroid, ax)
            else:
                self.centroid = gravity_center

            # Visualization
            self.vis.draw_current_detection(players_in_bev, ax)
            self.vis.draw_tracked_objects(active_tracks, ax)
            self.vis.draw_gravity_center(gravity_center, ax)
            
            self.data.save_result_img(plt)
            
            # Debug - Time
            """ self.end_time = time.time()
            print("Full cycle in sec: ",self.end_time - self.start_time)
            self.start_time = time.time() """
            
            plt.cla()
            plt.close()
        
        self.data.create_and_save_gif()

        print("GIF saved as animation.gif!")
        

if __name__ == "__main__":
    
    args = get_args()
    
    local_bev = BEV(args)
    local_bev.bev_main()

