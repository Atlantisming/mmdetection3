from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, OptMultiConfig


@MODELS.register_module()
class V2LBBoxHead(BBoxHead):

    def __init__(self,
                 embedding_based: bool = True,
                 emb_dim: int = 768,
                 cls_weight_path: str = './datasets/zeroshot_coco\
            /zero-shot/class_weight.npy',
                 freeze_emb_pred: bool = True,
                 with_avg_pool: bool = True,
                 with_cls: bool = True,
                 with_reg: bool = True,
                 roi_feat_size: int = 7,
                 in_channels: int = 256,
                 num_classes: int = 49,
                 bbox_coder: ConfigType = dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 predict_box_type: str = 'hbox',
                 reg_class_agnostic: bool = False,
                 reg_decoded_bbox: bool = False,
                 reg_predictor_cfg: ConfigType = dict(type='Linear'),
                 cls_predictor_cfg: ConfigType = dict(type='Linear'),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            with_avg_pool=with_avg_pool,
            with_cls=with_cls,
            with_reg=with_reg,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            num_classes=num_classes,
            bbox_coder=bbox_coder,
            predict_box_type=predict_box_type,
            reg_class_agnostic=reg_class_agnostic,
            reg_decoded_bbox=reg_decoded_bbox,
            reg_predictor_cfg=reg_predictor_cfg,
            cls_predictor_cfg=cls_predictor_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            init_cfg=init_cfg)
        self.embedding_based = embedding_based
        if self.embedding_based:
            self.emb_dim = emb_dim
            self.emb_pred = nn.Linear(in_channels, self.emb_dim)
            nn.init.normal_(self.emb_pred.weight, mean=0, std=0.01)
            nn.init.constant_(self.emb_pred.bias, 0)

            # TODO 检查 num_bbox_reg_class = 2
            # num_bbox_reg_classes = 2

            self.num_classes = num_classes
            # TODO 源代码为 None
            device = self.emb_pred.weight.device
            cls_weight = np.load(cls_weight_path)
            cls_weight = torch.from_numpy(cls_weight)
            self.num_classes = cls_weight.shape[0] - 1
            # self.fc_cls = None
            self.fc_cls = nn.Linear(self.emb_dim, self.num_classes + 1)
            self.fc_cls.weight.data = torch.tensor(
                cls_weight, device=device, requires_grad=False)
            self.fc_cls.bias.data = torch.zeros_like(
                self.fc_cls.bias.data, requires_grad=False)

            if freeze_emb_pred:
                self.emb_pred.weight.requires_grad = False
                self.emb_pred.bias.requires_grad = False
        else:
            # TODO 生成方式不同[*, 4] [*, 8]
            # num_bbox_reg_classes = 1 if reg_class_agnostic \
            #     else num_classes
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

            nn.init.normal_(self.fc_cls.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_cls.bias, 0)

        box_dim = self.bbox_coder.encode_size
        # 源码使用后 box_dim*2 使用后4个值做运算
        out_dim_reg = box_dim * 2 if reg_class_agnostic else \
            box_dim * num_classes
        reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
        if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
            reg_predictor_cfg_.update(
                in_features=in_channels, out_features=out_dim_reg)
        self.fc_reg = MODELS.build(reg_predictor_cfg_)
        # self.fc_reg = nn.Linear(in_channels, num_bbox_reg_classes * 4)
        nn.init.normal_(self.fc_reg.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x: Tuple) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * 4.
        """
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        if self.embedding_based:
            cls_emb = self.emb_pred(x)
            cls_score = self.fc_cls(cls_emb)
        else:
            cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        # print('bbox_pred_forward', bbox_pred.shape)
        # exit()
        return cls_score, bbox_pred

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """
        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    bbox_pred = bbox_pred[:, -4:]
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred[:, -4:]
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            # print('roi', roi[..., 1:].shape)
            # print('bbox_pred', bbox_pred.shape)
            # exit()
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
