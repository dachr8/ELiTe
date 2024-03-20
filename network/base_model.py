import os
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from utils.metric_util import IoU
from utils.schedulers import cosine_schedule_with_warmup


class LightningBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_acc = Accuracy()
        self.train_acc_2d = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.val_iou = IoU(self.args['dataset_params'], compute_on_step=False)

        if self.args['submit_to_server']:
            self.submit_dir = os.path.dirname(self.args['checkpoint']) + '/submit_' + datetime.now().strftime(
                '%Y_%m_%d')
            with open(self.args['dataset_params']['label_mapping'], 'r') as stream:
                self.mapfile = yaml.safe_load(stream)

        self.ignore_label = self.args['dataset_params']['ignore_label']

        self.timer = None
        self.save = False

    def configure_optimizers(self):
        if self.args['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=self.args['train_params']["learning_rate"])
        elif self.args['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                        lr=self.args['train_params']["learning_rate"],
                                        momentum=self.args['train_params']["momentum"],
                                        weight_decay=self.args['train_params']["weight_decay"],
                                        nesterov=self.args['train_params']["nesterov"])
        else:
            raise NotImplementedError

        if self.args['train_params']["lr_scheduler"] == 'StepLR':
            lr_scheduler = StepLR(
                optimizer,
                step_size=self.args['train_params']["decay_step"],
                gamma=self.args['train_params']["decay_rate"]
            )
        elif self.args['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.args['train_params']["decay_rate"],
                patience=self.args['train_params']["decay_step"],
                verbose=True
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args['train_params']['max_num_epochs'] - 4,
                eta_min=1e-5,
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
            from functools import partial
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.args['train_params']['max_num_epochs'],
                    batch_size=self.args['dataset_params']['train_data_loader']['batch_size'],
                    dataset_size=self.args['dataset_params']['training_size'],
                    num_gpu=len(self.args.gpu)
                ),
            )
        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' if self.args['train_params'][
                                      "lr_scheduler"] == 'CosineAnnealingWarmRestarts' else 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.args.monitor,
        }

    def forward(self, data):
        pass

    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        self.train_acc(data_dict['logits'].argmax(1)[data_dict['labels'] != self.ignore_label],
                       data_dict['labels'][data_dict['labels'] != self.ignore_label])
        self.log('train/acc', self.train_acc, on_epoch=True)
        self.log('train/loss', data_dict['loss'])
        self.log('train/loss_main_ce', data_dict['loss_main_ce'])
        self.log('train/loss_main_lovasz', data_dict['loss_main_lovasz'])

        if not self.baseline_only:
            if 'img_pseudo_label' in data_dict:
                img_label = data_dict['img_pseudo_label'].reshape(-1)
            else:
                img_label = data_dict['img_label']

            self.train_acc_2d(data_dict['logits_2d'].argmax(1)[img_label != self.ignore_label],
                              img_label[img_label != self.ignore_label])
            self.log('train/acc_2d', self.train_acc_2d, on_epoch=True)
            for scale in self.args['model_params']['scale_list']:
                self.log('train/loss_3d_scale{}'.format(scale), data_dict['loss_3d_scale{}'.format(scale)])
                self.log('train/loss_2d_scale{}'.format(scale), data_dict['loss_2d_scale{}'.format(scale)])
                self.log('train/loss_xm_scale{}'.format(scale), data_dict['loss_xm_scale{}'.format(scale)])
            self.log('train/loss_2d_multiscale', data_dict['loss_2d_multiscale'])
            if self.args['model_params']['image_encoder'].get('peft', False) == 'adalora':
                self.log('train/loss_2d_orth_regu', data_dict['loss_2d_orth_regu'])

        return data_dict['loss']

    def validation_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        prediction = data_dict['logits'].cpu().argmax(1)
        raw_labels = data_dict['labels'].squeeze(0).cpu()

        self.val_acc(prediction, raw_labels)
        self.log('val/acc', self.val_acc, on_epoch=True)
        self.val_iou(prediction.detach().numpy(), raw_labels.detach().numpy())

        return data_dict['loss']

    def test_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        origin_len = data_dict['origin_len']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        path = data_dict['path'][0]

        vote_logits = torch.zeros((len(raw_labels), self.num_classes))

        if self.timer is not None:
            import time
            torch.cuda.synchronize()
            start = time.time()
            data_dict = self.forward(data_dict)
            torch.cuda.synchronize()
            end = time.time()
            self.timer += end - start
            print(self.timer)
        else:
            data_dict = self.forward(data_dict)

        vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())

        if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
            vote_logits = vote_logits[:origin_len]
            raw_labels = raw_labels[:origin_len]

        prediction = vote_logits.argmax(1)

        if not self.args['submit_to_server']:
            self.val_acc(prediction, raw_labels)
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(prediction.cpu().detach().numpy(), raw_labels.cpu().detach().numpy())

            if self.save:
                from utils.helper_ply import write_ply
                a = Accuracy()
                acc = int(a(prediction, raw_labels).item() * 100)
                if acc < 90:
                    os.makedirs('logs/vis/2dpass', exist_ok=True)
                    path = data_dict['path'][0].replace('dataset/sequences',
                                                        '2DPASS-main/logs/vis/2dpass').replace('/velodyne/',
                                                                                               '_').replace(
                        '.bin', '_{}.ply'.format(acc))
                    write_ply(path, [data_dict['ref_xyz'].cpu().detach().numpy(), (
                            data_dict['logits'].argmax(1) != data_dict['labels']).cpu().detach().numpy().astype(
                        'int32')], ['x', 'y', 'z', 'error'])

        else:
            components = path.split('/')
            sequence = components[-3]
            points_name = components[-1]
            label_name = points_name.replace('bin', 'label')
            full_save_dir = os.path.join(self.submit_dir, 'sequences', sequence, 'predictions')
            os.makedirs(full_save_dir, exist_ok=True)
            full_label_name = os.path.join(full_save_dir, label_name)

            if os.path.exists(full_label_name):
                print('%s already exsist...' % label_name)
                pass

            valid_labels = np.vectorize(self.mapfile['learning_map_inv'].__getitem__)
            original_label = valid_labels(vote_logits.argmax(1).cpu().numpy().astype(int))
            final_preds = original_label.astype(np.uint32)
            final_preds.tofile(full_label_name)

        return data_dict['loss']

    def validation_epoch_end(self, outputs):
        iou, best_miou = self.val_iou.compute()
        mIoU = np.nanmean(iou)
        self.log('val/mIoU', mIoU, on_epoch=True)
        self.log('val/best_miou', best_miou, on_epoch=True)

        str_print = 'Validation per class iou: '
        for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
        self.print(str_print)

    def test_epoch_end(self, outputs):
        if self.timer is not None:
            self.print("inference time:", self.timer)

        if not self.args['submit_to_server']:
            iou, best_miou = self.val_iou.compute()
            mIoU = np.nanmean(iou)
            self.log('val/mIoU', mIoU, on_epoch=True)
            self.log('val/best_miou', best_miou, on_epoch=True)

            str_print = 'Validation per class iou: '
            for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
                str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

            str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
            self.print(str_print)

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

        if self.args['model_params']['backbone_2d'] != 'resnet34':
            if self.args['model_params']['image_encoder']['peft'] == 'adalora':
                self.rankallocator.update_and_mask(self.model_2d, self.global_step)
