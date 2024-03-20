import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from network.basic_block import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.segment_anything.modeling import ImageEncoderViT


class xModalKD(nn.Module):
    def __init__(self, config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.hiden_2d_size = config['model_params']['image_encoder']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)
        self.config = config

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, self.hiden_size * 2),
                    nn.ReLU(True),
                    nn.Linear(self.hiden_size * 2, self.num_classes))
            )

        self.multihead_2d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_2d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_2d_size, self.hiden_2d_size * 2),
                    nn.ReLU(True),
                    nn.Linear(self.hiden_2d_size * 2, self.num_classes))
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_2d_size * self.num_scales, self.hiden_2d_size * 2),
            nn.ReLU(True),
            nn.Linear(self.hiden_2d_size * 2, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights,
                                           ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max() + 1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def PPMSKD(self, data_dict, idx):
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        scale = self.scale_list[idx]
        img_feat = data_dict['img_scale{}'.format(scale)]
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']

        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

        # 3D loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        data_dict['loss_3d_scale{}'.format(scale)] = seg_loss_3d

        if 'img_pseudo_label' in data_dict:
            b, h, w = data_dict['img_pseudo_label'].shape

            # 2D prediction
            img_feat = img_feat.permute(0, 2, 3, 1).reshape(b * h * w, self.hiden_2d_size)
            img_pred = self.multihead_2d_classifier[idx](img_feat)

            # 2D loss
            seg_loss_2d = self.seg_loss(img_pred,
                                        data_dict['img_pseudo_label'].reshape(-1)) * self.lambda_seg2d / self.num_scales
            data_dict['loss_2d_scale{}'.format(scale)] = seg_loss_2d

            img_pred = img_pred.reshape(b, h, w, self.num_classes)
            img_indices = data_dict['img_indices']
            tmp = []
            for i in range(b):
                tmp.append(img_pred[i][img_indices[i][:, 0], img_indices[i][:, 1]])
            img_pred = torch.cat(tmp, 0)

        else:
            # 2D prediction
            img_pred = self.multihead_2d_classifier[idx](img_feat)

            # 2D loss
            seg_loss_2d = self.seg_loss(img_pred, data_dict['img_label']) * self.lambda_seg2d / self.num_scales
            data_dict['loss_2d_scale{}'.format(scale)] = seg_loss_2d

        # KL divergence
        xm_loss = F.kl_div(F.log_softmax(pts_pred, dim=1), F.softmax(img_pred.detach(), dim=1))
        data_dict['loss_xm_scale{}'.format(scale)] = xm_loss * self.lambda_xm / self.num_scales

        loss = data_dict['loss_3d_scale{}'.format(scale)] + data_dict['loss_2d_scale{}'.format(scale)] + data_dict[
            'loss_xm_scale{}'.format(scale)]

        return loss, img_feat

    def forward(self, data_dict):
        img_seg_feat = []

        for idx in range(self.num_scales):
            singlescale_loss, img_feat = self.PPMSKD(data_dict, idx)
            img_seg_feat.append(img_feat)
            data_dict['loss'] += singlescale_loss

        data_dict['logits_2d'] = self.classifier(torch.cat(img_seg_feat, 1))

        if 'img_pseudo_label' in data_dict:
            data_dict['loss_2d_multiscale'] = self.seg_loss(data_dict['logits_2d'],
                                                            data_dict['img_pseudo_label'].reshape(-1))
        else:
            data_dict['loss_2d_multiscale'] = self.seg_loss(data_dict['logits_2d'], data_dict['img_label'])
        data_dict['loss'] += data_dict['loss_2d_multiscale']

        '''
        tmp = torch.cat(img_seg_feat, 1)

        for i in range(len(self.classifier)):
            tmp = self.classifier[i](tmp)
            if i == 1:
                break

        np.save('logs/tsne/penultimate/elite/img_fea', tmp.cpu())
        '''

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)
        self.config = config

        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.model_2d = ImageEncoderViT(config,
                                            img_size=config.model_params.image_encoder.img_size,
                                            embed_dim=config.model_params.image_encoder.embed_dim,
                                            depth=config.model_params.image_encoder.depth,
                                            num_heads=config.model_params.image_encoder.num_heads,
                                            out_chans=config.model_params.hiden_size,
                                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                            use_rel_pos=True,
                                            window_size=14,
                                            global_attn_indexes=config.model_params.image_encoder.global_attn_indexes)
            self.fusion = xModalKD(config)

            if config.model_params.image_encoder.peft == 'adalora':
                from loralib import RankAllocator
                # Initialize the RankAllocator
                total_step = config.dataset_params.training_size * config.train_params.max_num_epochs // config.dataset_params.train_data_loader.batch_size // config.train_params.num_gpus
                self.rankallocator = RankAllocator(
                    self.model_2d,
                    lora_r=12,
                    target_rank=8,
                    init_warmup=int(total_step / 6),
                    final_warmup=int(total_step / 2),
                    mask_interval=int(total_step / 300),
                    total_step=total_step,
                    beta1=0.85,
                    beta2=0.85
                )
        else:
            print('Start vanilla training!')

    def forward(self, data_dict):
        # 3D network
        """
        data_dict = {
            'points'            = (Np, xyzi)
            'ref_xyz'           = (Np, xyz)
            'batch_idx'         = (Np,) in range(BS)
            'batch_size'        = BS
            'labels'            = (Np,) in range(C)
            'raw_labels'        = (OL, 1) in range(C)
            'origin_len'        = OL
            'indices'           = (Np,) in range(OL)
            'point2img_index'   = BS * (ni) in range(np)
            'img'               = (BS, rgb, H, W)
            'img_indices'       = BS * (ni, 2) in [range(H), range(W)]
            'img_label'         = (Ni,) in range(C)
            'path'              = BS
        """
        data_dict = self.model_3d(data_dict)
        """
        data_dict += {
            'scale_2': {
                'full_coors'    = (Np, 4)
                'coors_inv'     = (Np,)
                'coors'         = (Np_down2, 4)
            }
            'scale_4': {
                'full_coors'    = (Np, 4)
                'coors_inv'     = (Np,)
                'coors'         = (Np_down3, 4)
            }
            'scale_8': {
                'full_coors'    = (Np, 4)
                'coors_inv'     = (Np,)
                'coors'         = (Np_down4, 4)
            }
            'scale_16': {
                'full_coors'    = (Np, 4)
                'coors_inv'     = (Np,)
                'coors'         = (Np_down5, 4)
            }
            'scale_1': {
                'full_coors'    = (Np, 4)
                'coors_inv'     = (Np,)
                'coors'         = (Np_down1, 4)
            }
            'sparse_tensor'     = {SparseConvTensor}(Np_down5, hiden_size)
            'coors'             = (Np_down5, 4)
            'coors_inv'         = (Np,)
            'full_coors'        = (Np, 4)
            'layer_0': {
                'pts_feat'      = (Np_down1, hiden_size)
                'full_coors'    = (Np, 4)
            }
            'layer_1': {
                'pts_feat'      = (Np_down2, hiden_size)
                'full_coors'    = (Np, 4)
            }
            'layer_2': {
                'pts_feat'      = (Np_down3, hiden_size)
                'full_coors'    = (Np, 4)
            }
            'layer_3': {
                'pts_feat'      = (Np_down4, hiden_size)
                'full_coors'    = (Np, 4)
            }
            'logits'            = (Np, C)
            'loss'              = L
            'loss_main_ce'      = LMC
            'loss_main_lovasz'  = LML
        }
        """

        # training with 2D network
        if self.training and not self.baseline_only:
            data_dict = self.model_2d(data_dict)
            """
            img                 = (BS, in_chans, img_size, img_size)
            img_indices         = BS * (ni, 2) in [range(H), range(W)]

            data_dict += {
                'img_scale2'   = (Ni, hiden_size)
                'img_scale4'   = (Ni, hiden_size)
                'img_scale8'   = (Ni, hiden_size)
                'img_scale16'  = (Ni, hiden_size)
            """
            data_dict = self.fusion(data_dict)

            if self.config.model_params.image_encoder.peft == 'adalora':
                from loralib import compute_orth_regu
                data_dict['loss_2d_orth_regu'] = compute_orth_regu(self.model_2d, regu_weight=0.1)
                data_dict['loss'] += data_dict['loss_2d_orth_regu']

        return data_dict
