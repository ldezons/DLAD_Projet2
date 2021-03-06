import torch
import torch.nn.functional as F
from typing import Type, Any,  Union, List
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.SE = SqueezeAndExcitation(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.SE(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = _resnet_(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        int_ch = 256
        skip_4x_out_ch = 48
        self.conv_skip = ASPPpart(in_channels=skip_4x_ch, out_channels=skip_4x_out_ch, kernel_size=1)
        self.features_conv = torch.nn.Sequential(
            ASPPpart(in_channels=skip_4x_out_ch + bottleneck_ch, out_channels=int_ch, kernel_size=3, padding=1),
            ASPPpart(in_channels=int_ch, out_channels=int_ch, kernel_size=3, padding=1)
        )
        self.pred_conv = torch.nn.Sequential(
            ASPPpart(in_channels=skip_4x_out_ch + bottleneck_ch, out_channels=int_ch, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels=int_ch, out_channels=num_out_ch, kernel_size=3, padding=1)
        )

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        low_features = self.conv_skip(features_skip_4x)
        features_4x = F.interpolate(features_bottleneck, size=features_skip_4x.shape[2:],
                                    mode='bilinear', align_corners=False)
        features_4x = torch.cat([features_4x, low_features], dim=1)
        # Get the prediction from the concatenated features , we do not use barch normalization and ReLu as we are supposed to make predictions
        predictions_4x = self.pred_conv(features_4x)
        # Get the upsampled features, here we need to use barch normalization and ReLu and upsample the feature
        features_4x = self.features_conv(features_4x)
        features_4x = F.interpolate(features_4x, size=features_skip_4x.shape[2:],
                                    mode='bilinear', align_corners=False)

        return predictions_4x, features_4x


class ClassicalDecoder(torch.nn.Module):
    def __init__(self, bottleneck_ch, num_out_ch):
        super(ClassicalDecoder, self).__init__()
        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        features_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        predictions_4x = self.features_to_predictions(features_4x)
        return predictions_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):

        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()
        # ASPP implementation as shown in the figure
        self.conv1x1 = ASPPpart(in_channels, out_channels, kernel_size=1)
        self.conv3x3_rate6 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0],
                                      dilation=rates[0])
        self.conv3x3_rate12 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1],
                                       dilation=rates[1])
        self.conv3x3_rate18 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2],
                                       dilation=rates[2])
        #Pooling Has to be followed by a convolution as we want to extract features
        self.out_pooling = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                               ASPPpart(in_channels, out_channels, kernel_size=1))

        self.conv_final = ASPPpart(out_channels * 5, out_channels, kernel_size=1)


    def forward(self, x):
        #Create a list with all the modules as they are computed in parallel
        output = [self.conv1x1(x)]
        output.append(self.conv3x3_rate6(x))
        output.append(self.conv3x3_rate12(x))
        output.append(self.conv3x3_rate18(x))
        #Due to a shape error, we had to rescale the avg output
        avg = self.out_pooling(x)
        avg = F.interpolate(avg, size=output[0].shape[2:], mode='bilinear', align_corners=False)
        #Output
        output.append(avg)
        #Last Convulational layer
        return self.conv_final(torch.cat(output,dim=1))


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed

class OrdinalRegression(torch.nn.Module):
    def __init__(self):
        super(OrdinalRegression, self).__init__()
    def forward(self, x):
        N, C, H, W = x.size()
        ord_num = C // 2
        x = x.view(-1, 2, ord_num, H, W)
        if self.training:
            prob = F.log_softmax(x, dim=1).view(N, C, H, W)
            return prob

        ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        ord_label = torch.sum((ord_prob > 0.5), dim=1)
        return ord_prob, ord_label


def _resnet_(
    arch: str,
    block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> resnet.ResNet:
    model = resnet.ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model