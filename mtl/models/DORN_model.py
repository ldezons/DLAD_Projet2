import torch
import torch.nn.functional as F
from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, SelfAttention, ClassicalDecoder, SqueezeAndExcitation, OrdinalRegression
from mtl.datasets.definitions import MOD_SEMSEG, MOD_DEPTH
import numpy as np

class Distillation_with_DORN(torch.nn.Module):
    def __init__(self, cfg, outputs_desc, ord_num=90, gamma=1.0, beta=80.0):
        super().__init__()
        ch_out_seg = outputs_desc[MOD_SEMSEG]
        ch_out_depth = outputs_desc[MOD_DEPTH]
        ch_inter = 256

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        #Self Attention Module
        self.self_att = SelfAttention(ch_inter, ch_inter)
        #ASPP
        self.aspp = ASPP(ch_out_encoder_bottleneck, 256)
        #Squeeze and excitation
        self.SE = SqueezeAndExcitation(channels=ch_out_encoder_bottleneck)
        #Semantic Segmentation
        self.decoder_seg = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_seg)
        self.decoder_noskip_seg = ClassicalDecoder(ch_inter, ch_out_seg)
        #Depth Estimation
        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_depth)
        self.decoder_noskip_depth = ClassicalDecoder(ch_inter, ch_out_depth)
        #Ordinal regression
        self.Ordinalregression = OrdinalRegression()

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        #print(", ".join([f"{k}:{v.shape[2:]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]
        features_out = self.aspp(features_lowest)

        out = F.interpolate(features_out, size=input_resolution, mode="bilinear", align_corners=True)
        prob, label = self.Ordinalregression(out)
        t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
        t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        depth = (t0 + t1) / 2 - self.gamma

        predictions_4x_seg, features_4x_seg = self.decoder_seg(features_out, features[4])


        predictions_1x_seg = F.interpolate(predictions_4x_seg, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_depth = F.interpolate(depth, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        out[MOD_SEMSEG] = predictions_1x_seg
        out[MOD_DEPTH] = predictions_1x_depth

        return out