from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


video_fpath = 'SwinBERT/test_mp4/atwt-01-01-04.mp4'

sm=SceneManager()
sm.add_detector(ContentDetector(threshold=100))
video = open_video(video_fpath)
sm.detect_scenes(video,show_progress=True)

scenes = sm.get_scene_list()

frame_itererator = cv2.VideoCapture(video_fpath)

down_sample_rate = 2
np_scene_frames = []

for i, scene in enumerate(scenes):
    breakpoint()
    starttime = scene[0].frame_num/video.frame_rate
    endtime = scene[1].frame_num/video.frame_rate
    ffmpeg_extract_subclip(video_fpath,starttime, endtime, targetname=f'scene{i}.mp4')
