import torch
import time
from torch import nn
#from ..decode_heads.lpls_utils import Lap_Pyramid_Conv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.runner import auto_fp16

@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 down_ratio,
                 backbone,
                 decode_head,
                 refine_input_ratio=1.,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 is_frequency=False,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        self.is_frequency = is_frequency
        self.down_scale = down_ratio
        self.refine_input_ratio = refine_input_ratio
        super(EncoderDecoderRefine, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        # self.decode_head = nn.ModuleList()
        #self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=1)
        self.decode_head = builder.build_head(decode_head[0])
        self.refine_head = builder.build_head(decode_head[1])
        # print(self.decode_head)
        # print(self.refine_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # TODO: 下采样图像设定尚未完成
        # 这里值得注意的是，输入图像应该分大分辨率和小分辨率两种
        # 目前的计划：
        # 大分辨率图像：参数传入的image， 输入refine_head中
        # 小分辨率图像：参数传入的image下采样到原来的0.25倍数，输入feature_extractor, 即原有的分支中
        if self.is_frequency:
            deeplab_inputs = self.lap_prymaid_conv.pyramid_decom(img)[0]
            img_os2 = nn.functional.interpolate(deeplab_inputs, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        else:
            img_os2 = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        
        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        # torch.cuda.synchronize()
        # start_time1 = time.perf_counter()
        x = self.extract_feat(img_os2)
        # 这里先假设就是只有一个decoder，每个decoder应该返回一组feature map或者是list
        # fm_decoder是decode返回的特征图，或者是一个特征图的list
        out_g, prev_outputs = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        # torch.cuda.synchronize()
        # end_time1 = time.perf_counter()
        # print(end_time1 - start_time1)
        # refine_head的输出作为最后的输出特征图
        # 其实这里的refine_head承担了两个功能，第一个是提取大尺度分辨率，第二是融合spatial feature map和context feature map
        # torch.cuda.synchronize()
        # start_time2 = time.perf_counter()
        out = self.refine_head.forward_test(img_refine, prev_outputs, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # torch.cuda.synchronize()
        # end_time2 = time.perf_counter()
        # print(end_time2 - start_time2)
        # print("--------------------------")
        return out
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # img_os2:将deeplabv3输入的图像size下采样为原来的一半
        if self.is_frequency:
            deeplab_inputs = self.lap_prymaid_conv.pyramid_decom(img)[0]
            img_os2 = nn.functional.interpolate(deeplab_inputs, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        else:
            img_os2 = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        x = self.extract_feat(img_os2)
        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_refine, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO: 搭建refine的head
    def _decode_head_forward_train(self, x, img, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, prev_features = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        # loss_contrsative_list = self.refine_head.forward_train(
        #          img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        # losses.update(add_prefix(loss_contrsative_list, 'second_'))loss_refine_aux16, loss_refine_aux8,
        loss_refine, *loss_contrsative_list = self.refine_head.forward_train(img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_refine, 'refine'))
        # losses.update(add_prefix(loss_refine_aux16, 'refine_aux16'))
        # losses.update(add_prefix(loss_refine_aux8, 'refine_aux8'))
        j = 1
        for loss_aux in loss_contrsative_list:
            losses.update(add_prefix(loss_aux, 'aux_' + str(j)))
            j += 1
        return losses
