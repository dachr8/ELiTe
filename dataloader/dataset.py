"""
Task-specific Datasets
"""
import random
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms as T

REGISTERED_DATASET_CLASSES = {}
REGISTERED_COLATE_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def register_collate_fn(cls, name=None):
    global REGISTERED_COLATE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_COLATE_CLASSES, f"exist class: {REGISTERED_COLATE_CLASSES}"
    REGISTERED_COLATE_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def get_collate_class(name):
    global REGISTERED_COLATE_CLASSES
    assert name in REGISTERED_COLATE_CLASSES, f"available class: {REGISTERED_COLATE_CLASSES}"
    return REGISTERED_COLATE_CLASSES[name]


@register_dataset
class point_image_dataset_semkitti(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=None, max_dropout_ratio=0.2):
        """Initialization"""
        if trans_std is None:
            trans_std = [0.1, 0.1, 0.1]
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']

    def __len__(self):
        """Denotes the total number of samples"""
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def get_augment_scene(self, index, cut_scene=False):
        """Generates one sample of data"""
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        if cut_scene:
            mask *= instance_label != 0

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # load 2D data
        image = data['img']
        proj_matrix = data['proj_matrix']

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]

        # 3D Augmentation
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        img_label = labels[keep_idx]
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        # 2D Augmentation
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        data_dict = {
            'point_feat': feat,
            'point_label': labels,
            'ref_xyz': ref_pc,
            'ref_label': ref_labels,
            'ref_index': ref_index,
            'mask': mask,
            'point_num': point_num,
            'origin_len': origin_len,
            'root': root,
            'img': image,
            'img_indices': img_indices,
            'img_label': img_label,
            'point2img_index': point2img_index
        }

        return data_dict

    def __getitem__(self, index):
        data_dict = self.get_augment_scene(index)

        if self.point_cloud_dataset.imageset == 'train' and self.config.dataset_params.get('cutmix', False):
            cut_index = random.randint(0, self.__len__() - 1)

            while cut_index == index:
                cut_index = random.randint(0, self.__len__() - 1)

            cut_dict = self.get_augment_scene(cut_index, cut_scene=True)
            cutmix_data_dict = {}
            for keys in data_dict.keys():
                if keys == 'point_num' or keys == 'origin_len':
                    cutmix_data_dict[keys] = data_dict[keys] + cut_dict[keys]
                elif keys == 'ref_index':
                    cut_dict[keys] = cut_dict[keys] + data_dict['origin_len']
                    cutmix_data_dict[keys] = np.append(data_dict[keys], cut_dict[keys])
                elif keys == 'mask':
                    cutmix_data_dict[keys] = np.append(data_dict[keys], cut_dict[keys])
                elif keys not in ['img', 'img_indices', 'img_label', 'point2img_index']:
                    cutmix_data_dict[keys] = np.vstack((data_dict[keys], cut_dict[keys]))
                else:
                    cutmix_data_dict[keys] = data_dict[keys]
        else:
            cutmix_data_dict = data_dict

        return cutmix_data_dict


@register_dataset
class point_image_dataset_semkitti_sam(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=None, max_dropout_ratio=0.2):
        """Initialization"""
        if trans_std is None:
            trans_std = [0.1, 0.1, 0.1]
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']

    def __len__(self):
        """Denotes the total number of samples"""
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def get_augment_scene(self, index, cut_scene=False):
        """Generates one sample of data"""
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        if cut_scene:
            mask *= instance_label != 0

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # load 2D data
        image = data['img']
        proj_matrix = data['proj_matrix']
        if self.point_cloud_dataset.pseudo_label:
            pseudo_semantic_label = data['pseudo_semantic_label']
        else:
            pseudo_semantic_label = None

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]

        # 3D Augmentation
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        if pseudo_semantic_label is None:
            img_label = labels[keep_idx]
        else:
            img_label = None
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        # 2D Augmentation
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            if pseudo_semantic_label is not None:
                pseudo_semantic_label = pseudo_semantic_label[top:bottom, left:right]
            else:
                img_label = img_label[keep_idx]
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # PIL to numpy
        image = np.array(image, copy=False)

        # 2D augmentation
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
            if pseudo_semantic_label is not None:
                pseudo_semantic_label = np.ascontiguousarray(np.fliplr(pseudo_semantic_label))

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        # Pad
        h, w = image.shape[:2]
        padh = self.config.model_params.image_encoder.img_size - h
        padw = self.config.model_params.image_encoder.img_size - w
        image = np.pad(image, [(0, padh), (0, padw), (0, 0)], mode='constant')

        data_dict = {
            'point_feat': feat,  # (PN, xyzi)
            'point_label': labels,  # (PN, 1) in range(NC)
            'ref_xyz': ref_pc,  # (PN, xyz)
            'ref_label': ref_labels,  # (OL, 1) in range(NC)
            'ref_index': ref_index,  # (PN,) in range(OL)
            'mask': mask,  # (PN,) in {0, 1}
            'point_num': point_num,  # {int} PN
            'origin_len': origin_len,  # {int} OL
            'root': root,  # {str} R
            'img': image,  # (H, W, rgb)
            'img_indices': img_indices,  # (PI, 2)  in [range(H), range(W)]
            'point2img_index': point2img_index  # (PI,) in range(PN)
        }

        if pseudo_semantic_label is not None:
            data_dict['img_pseudo_label'] = pseudo_semantic_label  # (H, W)
        if img_label is not None:
            data_dict['img_label'] = img_label  # (PI, 1) in range(NC)

        return data_dict

    def __getitem__(self, index):
        data_dict = self.get_augment_scene(index)

        if self.point_cloud_dataset.imageset == 'train' and self.config.dataset_params.get('cutmix', False):
            cut_index = random.randint(0, self.__len__() - 1)

            while cut_index == index:
                cut_index = random.randint(0, self.__len__() - 1)

            cut_dict = self.get_augment_scene(cut_index, cut_scene=True)
            cutmix_data_dict = {}
            for keys in data_dict.keys():
                if keys == 'point_num' or keys == 'origin_len':
                    cutmix_data_dict[keys] = data_dict[keys] + cut_dict[keys]
                elif keys == 'ref_index':
                    cut_dict[keys] = cut_dict[keys] + data_dict['origin_len']
                    cutmix_data_dict[keys] = np.append(data_dict[keys], cut_dict[keys])
                elif keys == 'mask':
                    cutmix_data_dict[keys] = np.append(data_dict[keys], cut_dict[keys])
                elif keys not in ['img', 'img_indices', 'img_label', 'img_pseudo_label', 'point2img_index']:
                    cutmix_data_dict[keys] = np.vstack((data_dict[keys], cut_dict[keys]))
                else:
                    cutmix_data_dict[keys] = data_dict[keys]
        else:
            cutmix_data_dict = data_dict

        return cutmix_data_dict


@register_collate_fn
def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    point2img_index = [torch.from_numpy(d['point2img_index']).long() for d in data]
    path = [d['root'] for d in data]

    img = [torch.from_numpy(d['img']) for d in data]
    img_indices = [d['img_indices'] for d in data]

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]

    data_dict = {
        'points': torch.cat(points).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'point2img_index': point2img_index,
        'img': torch.stack(img, 0).permute(0, 3, 1, 2),
        'img_indices': img_indices,
        'path': path,
    }

    if 'img_label' in data[0]:
        img_label = [torch.from_numpy(d['img_label']) for d in data]
        data_dict['img_label'] = torch.cat(img_label, 0).squeeze(1)
    if 'img_pseudo_label' in data[0]:
        img_pseudo_label = [torch.from_numpy(d['img_pseudo_label']) for d in data]
        data_dict['img_pseudo_label'] = torch.stack(img_pseudo_label)

    return data_dict


@register_collate_fn
def collate_fn_plg(data):
    return {
        'img': [d['img'] for d in data],
        'img_indices': [d['img_indices'] for d in data],
        'img_label': [d['img_label'] for d in data],
        'instance_label': [d['instance_label'] for d in data],
        'path': [d['root'] for d in data]
    }
