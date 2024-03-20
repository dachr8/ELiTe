import os
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.metrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from utils.metric_util import IoU
from utils.schedulers import cosine_schedule_with_warmup


class LightningBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.val_iou = IoU(self.args['dataset_params'], compute_on_step=False)

        if self.args['submit_to_server']:
            self.submit_dir = os.path.dirname(self.args['checkpoint']) + '/submit_' + datetime.now().strftime(
                '%Y_%m_%d')
            with open(self.args['dataset_params']['label_mapping'], 'r') as stream:
                self.mapfile = yaml.safe_load(stream)

        self.ignore_label = self.args['dataset_params']['ignore_label']

    def configure_optimizers(self):
        if self.args['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.args['train_params']["learning_rate"])
        elif self.args['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
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
        self.log('train/loss_main_ce', data_dict['loss_main_ce'])
        self.log('train/loss_main_lovasz', data_dict['loss_main_lovasz'])

        return data_dict['loss']

    def validation_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        origin_len = data_dict['origin_len']
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        data_dict = self.forward(data_dict)

        if self.args['test']:
            vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())
            if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
                vote_logits = vote_logits[:origin_len]
                raw_labels = raw_labels[:origin_len]
        else:
            vote_logits = data_dict['logits'].cpu()
            raw_labels = data_dict['labels'].squeeze(0).cpu()

        prediction = vote_logits.argmax(1)

        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1

        self.val_acc(prediction, raw_labels)
        self.log('val/acc', self.val_acc, on_epoch=True)
        self.val_iou(
            prediction.cpu().detach().numpy(),
            raw_labels.cpu().detach().numpy(),
        )

        return data_dict['loss']

    def test_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        origin_len = data_dict['origin_len']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        path = data_dict['path'][0]

        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        data_dict = self.forward(data_dict)
        vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())

        if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
            vote_logits = vote_logits[:origin_len]
            raw_labels = raw_labels[:origin_len]

        prediction = vote_logits.argmax(1)

        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1

        if not self.args['submit_to_server']:
            self.val_acc(prediction, raw_labels)
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(
                prediction.cpu().detach().numpy(),
                raw_labels.cpu().detach().numpy(),
            )
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
        str_print = ''
        self.log('val/mIoU', mIoU, on_epoch=True)
        self.log('val/best_miou', best_miou, on_epoch=True)
        str_print += 'Validation per class iou: '

        for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
        self.print(str_print)

    def test_epoch_end(self, outputs):
        if not self.args['submit_to_server']:
            iou, best_miou = self.val_iou.compute()
            mIoU = np.nanmean(iou)
            str_print = ''
            self.log('val/mIoU', mIoU, on_epoch=True)
            self.log('val/best_miou', best_miou, on_epoch=True)
            str_print += 'Validation per class iou: '

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
