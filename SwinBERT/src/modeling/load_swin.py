import torch
from src.modeling.video_swin.swin_transformer import SwinTransformer3D
from src.modeling.video_swin.config import Config

def get_swin_model(img_res, vidswin_size, kinetics, pretrained_2d, use_grid_feat):
    if int(img_res) == 384:
        assert vidswin_size == "large"
        config_path = 'SwinBERT/src/modeling/video_swin/swin_%s_384_patch244_window81212_kinetics%s_22k.py'%(vidswin_size, kinetics)
        model_path = './SwinBERT/models/video_swin_transformer/swin_%s_384_patch244_window81212_kinetics%s_22k.pth'%(vidswin_size, kinetics)
    else:
        # in the case that img_res == '224'
        config_path = 'SwinBERT/src/modeling/video_swin/swin_%s_patch244_window877_kinetics%s_22k.py'%(vidswin_size, kinetics)
        model_path = './SwinBERT/models/video_swin_transformer/swin_%s_patch244_window877_kinetics%s_22k.pth'%(vidswin_size, kinetics)
    if pretrained_2d:
        config_path = 'SwinBERT/src/modeling/video_swin/swin_base_patch244_window877_kinetics400_22k.py'
        model_path = './SwinBERT/models/swin_transformer/swin_base_patch4_window7_224_22k.pth'

    cfg = Config.fromfile(config_path)
    pretrained_path = model_path if pretrained_2d else None
    backbone = SwinTransformer3D(
                    pretrained=pretrained_path,
                    pretrained2d=pretrained_2d,
                    patch_size=cfg.model['backbone']['patch_size'],
                    in_chans=3,
                    embed_dim=cfg.model['backbone']['embed_dim'],
                    depths=cfg.model['backbone']['depths'],
                    num_heads=cfg.model['backbone']['num_heads'],
                    window_size=cfg.model['backbone']['window_size'],
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.2,
                    norm_layer=torch.nn.LayerNorm,
                    patch_norm=cfg.model['backbone']['patch_norm'],
                    frozen_stages=-1,
                    use_checkpoint=False)

    video_swin = myVideoSwin(backbone=backbone, use_grid_feat=use_grid_feat)

    if not pretrained_2d:
        checkpoint_3d = torch.load(model_path, map_location='cpu')
        video_swin.load_state_dict(checkpoint_3d['state_dict'], strict=False)
    else:
        video_swin.backbone.init_weights()
    return video_swin

def reload_pretrained_swin(video_swin, args):
    if not args.reload_pretrained_swin:
        return video_swin
    if int(args.img_res) == 384:
        model_path = './models/video_swin_transformer/swin_%s_384_patch244_window81212_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)
    else:
        # in the case that args.img_res == '224'
        model_path = './models/video_swin_transformer/swin_%s_patch244_window877_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)

    checkpoint_3d = torch.load(model_path, map_location='cpu')
    missing, unexpected = video_swin.load_state_dict(checkpoint_3d['state_dict'], strict=False)
    return video_swin

class myVideoSwin(torch.nn.Module):
    def __init__(self, backbone, use_grid_feat):
        super(myVideoSwin, self).__init__()
        self.backbone = backbone
        self.use_grid_feature = use_grid_feat

    def forward(self, x):
        x = self.backbone(x)
        return x
