from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import time
#from team_assigner import TeamAssigner
#from player_ball_assigner import PlayerBallAssigner
#from camera_movement_estimator import CameraMovementEstimator
#from view_transformer import ViewTransformer
#from speed_and_distance_estimator import SpeedAndDistance_Estimator



#def main():
#    video_frames = read_video(r'videos_for_cvat\test.mp4')
#
#    tracker = Tracker(r'from_kaggle\runs\detect\train\weights\best.pt')
#
#    tracks = tracker.get_object_track(video_frames,
#                                      read_from_stub=True,
#                                      stub_path=r'stubs\track_stubs.pkl')
#    
#
#    output_video_frames = tracker.draw_annotations(video_frames, tracks)
#    
#    cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
#    cv2.resizeWindow("Output Video", 640, 360)  # Устанавливаем размер окна
#    fps = 24
#    for frame in output_video_frames:
#        cv2.imshow('Output Video', frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажатие 'q' для выхода
#            break
#        #time.sleep(1/fps)  # Ждем 1 секунды между кадрами
#
#    cv2.destroyAllWindows()  # Закрываем все окна
#
#    #save_video(output_video_frames, 'output_videos/output_video.avi')




def main():
    video_frames = read_video(r'videos_for_cvat\test.mp4')
    print(f"Количество кадров в видео: {len(video_frames)}")

    tracker = Tracker(r'from_kaggle\runs\detect\train\weights\best.pt')

    tracks = tracker.get_object_track(video_frames,
                                      read_from_stub=True,
                                      stub_path=r'stubs\track_stubs.pkl')
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    #cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Output Video", 640, 360)
    #fps = 24
    #
    #for frame in output_video_frames:
    #    cv2.imshow('Output Video', frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #    time.sleep(1/fps)
    #
    #cv2.destroyAllWindows()


    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()

