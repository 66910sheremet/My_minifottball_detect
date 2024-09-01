from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        logger.info(f"Начало обработки кадров. Кадров: {len(frames)}")

        detections = []
        for i in range(0, len(frames), 60):  # Используем batch_size=60
            batch_end = min(i + 60, len(frames))
            batch = frames[i:batch_end]
            
            logger.debug(f"Обработка порции кадров: {i}-{batch_end} из {len(frames)}")
            
            predictions = self.model.predict(batch, conf=0.1)
            
            detections.extend(predictions)
            
            if i % (len(frames) // 10) == 0:
                logger.info(f"Обработано {i//60*60}% кадров")

        logger.info(f"Завершение обработки кадров. Кадров: {len(frames)}")
        return detections

    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "person": [],
            "referee": [],
            "sports ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for obj_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[obj_ind] = cls_names_inv["person"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["person"].append({})
            tracks["referee"].append({})
            tracks["sports ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['person']:
                    tracks["person"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['sports ball']:
                    tracks["sports ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        logger.debug(f"Нарисование эллипса для bbox: {bbox}, цвет: {color}, track_id: {track_id}")

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(int(x_center), int(y2)),
            axes=(int(width), int(0.35 * width)),  # Увеличиваем размеры
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            thickness=2,
            color=color
        )

        return frame.copy()

    def draw_annotations(self, video_frames, tracks):
        logger.debug(f"Начало draw_annotations. Кадров: {len(video_frames)}")
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame_copy = frame.copy()

            person_dict = tracks["person"][frame_num] or {}
            ball_dict = tracks["sports ball"][frame_num] or {}
            referee_dict = tracks["referee"][frame_num] or {}

            logger.debug(f"Обработка кадра {frame_num}")
            
            for track_id, person in person_dict.items():
                frame_copy = self.draw_ellipse(frame_copy, person["bbox"], (0,0,255), track_id)

            output_video_frames.append(frame_copy)
            
            logger.debug(f"Завершение обработки кадра {frame_num}")
            
        print(f"Завершение draw_annotations. Кадров: {len(output_video_frames)}")
        return output_video_frames
