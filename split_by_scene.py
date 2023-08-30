from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from SwinBERT.src.tasks.run_caption_VidSwinBert_inference import inference
from SwinBERT.src.datasets.caption_tensorizer import build_tensorizer
from SwinBERT.src.modeling.load_swin import get_swin_model
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.datasets.data_utils.video_ops import extract_frames_from_video_path
import torch


video_fpath = 'test_mp4/atwt-01-01-04.mp4'
img_res = 224
max_num_frames = 64

sm=SceneManager()
sm.add_detector(ContentDetector(threshold=100))
video = open_video(video_fpath)
sm.detect_scenes(video,show_progress=True)

scenes = sm.get_scene_list()

frame_itererator = cv2.VideoCapture(video_fpath)

down_sample_rate = 2
np_scene_frames = []

bert_model, config, tokenizer = get_bert_model(do_lower_case=True)
swin_model = get_swin_model(img_res, 'base', '600', False, True)
vl_transformer = VideoTransformer(True, config, swin_model, bert_model)
vl_transformer.freeze_backbone(freeze=False)
pretrained_model = torch.load('SwinBERT/models/table1/vatex/best-checkpoint/model.bin', map_location=torch.device('cpu'))
vl_transformer.load_state_dict(pretrained_model, strict=False)
vl_transformer.cuda()
vl_transformer.eval()

tensorizer = build_tensorizer(tokenizer, 50, 1568, 20, is_train=False)

scenes_dir = 'atwt1_scenes'
caps_per_scene = []
for i, scene in enumerate(scenes):
    starttime = scene[0].frame_num/video.frame_rate
    endtime = scene[1].frame_num/video.frame_rate
    scene_vid_path = f'{scenes_dir}/scene{i}.mp4'
    ffmpeg_extract_subclip(video_fpath,starttime, endtime, targetname=scene_vid_path)
    frames, _ = extract_frames_from_video_path(
                scene_vid_path, target_fps=3, num_frames=64,
                multi_thread_decode=False, sampling_strategy="uniform",
                safeguard_duration=False, start=None, end=None)
    cap = inference(scene_vid_path, img_res, max_num_frames, vl_transformer, tokenizer, tensorizer)
    caps_per_scene.append(cap)

with open(f'{scenes_dir}/captions_per_scene.txt','w') as f:
    for i,c in enumerate(caps_per_scene):
        f.write(f'{i}: {c}')

