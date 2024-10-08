from __future__ import absolute_import, division, print_function
from time import time
import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, pythonpath)
import numpy as np
from PIL import Image
import os.path as op
import json
import torch
import torch.distributed as dist
from src.configs.config import basic_check_arguments, shared_configs
from src.datasets.data_utils.video_ops import extract_frames_from_video_path
from src.datasets.data_utils.video_transforms import Compose, Resize, Normalize, CenterCrop
from src.datasets.data_utils.volume_transforms import ClipToTensor
from src.datasets.caption_tensorizer import build_tensorizer
from src.utils.comm import dist_init
from src.utils.miscellaneous import (set_seed, str_to_bool)
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.modeling.load_swin import get_swin_model
from src.modeling.load_bert import get_bert_model

def _online_video_decode(decoder_num_frames, video_path):
    frames, _ = extract_frames_from_video_path(
                video_path, target_fps=3, num_frames=decoder_num_frames,
                multi_thread_decode=False, sampling_strategy="uniform",
                safeguard_duration=False, start=None, end=None)
    return frames

def _transforms(img_res, max_num_frames, frames):
    raw_video_crop_list = [
        Resize(img_res),
        CenterCrop((img_res,img_res)),
        ClipToTensor(channel_nb=3),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
    raw_video_prcoess = Compose(raw_video_crop_list)

    frames = frames.numpy()
    if frames.ndim == 5: # collapse time and batch dims
        was_batched = True
        bs, n_frames = frames.shape[:2]
        frames = frames.reshape(-1,*frames.shape[2:])
    else:
        assert frames.ndim == 4
        was_batched = False
        bs = 1
    frames = np.transpose(frames, (0, 2, 3, 1))

    frame_list = []
    N = min(bs*max_num_frames,frames.shape[0])
    for i in range(N):
        frame_list.append(Image.fromarray(frames[i]))

    # apply normalization, output tensor (C x T x H x W) in the range [0, 1.0]
    crop_frames = raw_video_prcoess(frame_list)
    # (C x T x H x W) --> (T x C x H x W)
    crop_frames = crop_frames.permute(1, 0, 2, 3)
    if was_batched:
        crop_frames = crop_frames.reshape(bs, n_frames, *crop_frames.shape[1:])
    return crop_frames

def inference(frames,img_res,max_num_frames, model, tokenizer, tensorizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])

    model.float()
    model.eval()
    preproc_frames = _transforms(img_res, max_num_frames, frames)
    X = tensorizer.tensorize_example_e2e('', preproc_frames)
    if frames.ndim==5:
        bs = len(frames)
        X = [x if i==3 else x.repeat(bs,*[1 for _ in range(x.ndim)]) for i,x in enumerate(X)]
    else:
        bs = 1
        X = [x[None] for x in X]
    X = tuple(t.cuda() for t in X)
    with torch.no_grad():

        inputs = {'is_decode': True,
            #'input_ids': X[0][None,:], 'attention_mask': X[1][None,:],
            'input_ids': X[0], 'attention_mask': X[1],
            #'token_type_ids': X[2][None,:], 'img_feats': X[3][None,:],
            'token_type_ids': X[2], 'img_feats': X[3],
            'masked_pos': X[4],
            'do_sample': False,
            'bos_token_id': cls_token_id,
            'pad_token_id': pad_token_id,
            'eos_token_ids': [sep_token_id],
            'mask_token_id': mask_token_id,
            'add_od_labels': False, # object-detection labels
            #'od_labels_start_posid': args.max_seq_a_length,
            # hyperparameters of beam search
            'max_length': tensorizer.max_seq_len,
            'num_beams': 1,
            "temperature": 1,
            "top_k": 0,
            "top_p": 1,
            "repetition_penalty": 1,
            "length_penalty": 1,
            "num_return_sequences": 1,
            "num_keep_best": 1,
        }
        #inputs = {'is_decode': True,
        #    'input_ids': X[0][None,:], 'attention_mask': X[1][None,:],
        #    'token_type_ids': X[2][None,:], 'img_feats': X[3][None,:],
        #    'masked_pos': X[4][None,:],
        #    'do_sample': False,
        #    'bos_token_id': cls_token_id,
        #    'pad_token_id': pad_token_id,
        #    'eos_token_ids': [sep_token_id],
        #    'mask_token_id': mask_token_id,
        #    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
        #    # hyperparameters of beam search
        #    'max_length': args.max_gen_length,
        #    'num_beams': args.num_beams,
        #    "temperature": args.temperature,
        #    "top_k": args.top_k,
        #    "top_p": args.top_p,
        #    "repetition_penalty": args.repetition_penalty,
        #    "length_penalty": args.length_penalty,
        #    "num_return_sequences": args.num_return_sequences,
        #    "num_keep_best": args.num_keep_best,
        #}
        #start_time = time()
        outputs = model(**inputs)
        #print(f'true inference time: {time()-start_time:.3f}')

        all_caps = outputs[0]  # batch_size * num_keep_best * max_len
        #all_confs = torch.exp(outputs[1])

        return [tokenizer.decode(all_caps[i,0,:].tolist(),skip_special_tokens=True) for i in range(bs)]

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames/2)*(int(args.img_res)/32)*(int(args.img_res)/32))

    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True

    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled==True:
        args.attn_mask_type = 'learn_vid_att'
def update_existing_config_for_inference(args):
    assert args.do_test or args.do_eval
    checkpoint = args.eval_model_dir
    try:
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)

        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_checkpoint = args.eval_model_dir + 'model.bin'
    train_args.model_name_or_path = 'SwinBERT/models/captioning/bert-base-uncased/'
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_test = True
    train_args.val_yaml = args.val_yaml
    train_args.video_dir = args.video_dir
    return train_args

def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--att_mask_expansion', type=int, default=-1, help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1, help="0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--video_dir', type=str, default='None')
    args = base_config.parse_args()
    return args

def main(args):
    args = update_existing_config_for_inference(args)
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    check_arguments(args)
    set_seed(args.seed, args.num_gpus)

     # Get Video Swin model
    swin_model = get_swin_model(args.img_res, args.vidswin_size, args.kinetics, args.pretrained_2d, args.grid_feat)
    # Get BERT and tokenizer
    bert_model, config, tokenizer = get_bert_model(args.do_lower_case)
    # build SwinBERT based on training configs
    vl_transformer = VideoTransformer(args.grid_feat, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    pretrained_model = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))

    if isinstance(pretrained_model, dict):
        vl_transformer.load_state_dict(pretrained_model, strict=False)
    else:
        vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    vl_transformer.to(args.device)
    vl_transformer.eval()

    tensorizer = build_tensorizer(tokenizer, args.max_seq_length, args.max_img_seq_length, args.max_gen_length, is_train=False)
    all_caps = {}
    for fn in os.listdir(args.video_dir):
        video_fpath = os.path.join(args.video_dir,fn)
        cap = inference(video_fpath, args.img_res, args.max_num_frames, vl_transformer, tokenizer, tensorizer)
        fn_ = fn.split('.')[0]
        all_caps[fn_] = cap
        print(f'{fn_}: {cap}')

    with open('all_vid_caps.json','w') as f:
        json.dump(all_caps,f)

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
