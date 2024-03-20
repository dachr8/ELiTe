import argparse
import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from collections import Counter

from dataloader.dataset import get_collate_class
from dataloader.pc_dataset import get_pc_model_class
from plg.segment_anything import sam_model_registry, SamPseudoLabelGenerator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-type",
    type=str,
    default='default',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default='../segment-anything/checkpoint/sam_vit_h_4b8939.pth',
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="The device to run generation on."
)

parser.add_argument(
    '--config_path',
    default='config/SAMPLG-semantickitti.yaml'
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def main(args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)
    config.update(vars(args))
    config = EasyDict(config)

    pc_dataset = get_pc_model_class(config.dataset_params.pc_dataset_type)
    train_config = config.dataset_params.train_data_loader

    dataset = pc_dataset(config, data_path=train_config['data_path'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             collate_fn=get_collate_class(config.dataset_params.collate_type))

    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device=args.device)
    amg_kwargs = get_amg_kwargs(args)
    # from plg.segment_anything import SamAutomaticMaskGenerator
    # generator = SamAutomaticMaskGenerator(sam, **amg_kwargs)
    generator = SamPseudoLabelGenerator(sam, **amg_kwargs)

    for data_dict in tqdm(dataloader):
        img = data_dict['img'][0]  # (H, W, rgb)
        img_indices = data_dict['img_indices'][0]  # (PI, 2)  in [range(H), range(W)]
        img_label = data_dict['img_label'][0]  # (PI, 1) in range(NC)
        instance_label = data_dict['instance_label'][0]  # (PI, 1) in range(NC)
        path = data_dict['path'][0]
        save_file = path.replace('sequences', 'processed_sk').replace('velodyne', 'image_2_labels').replace('.bin',
                                                                                                            '.npy')
        save_base = "/".join(save_file.split('/')[:-1])
        os.makedirs(save_base, exist_ok=True)
        if os.path.exists(save_file):
            print(f"Skip {save_file}")
        else:
            # masks = generator.generate(img)
            masks = generator.generate(img, img_indices, img_label, instance_label)

            """
            import matplotlib.pyplot as plt
            def show_mask(mask, ax, random_color=False):
                if random_color:
                    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                else:
                    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                ax.imshow(mask_image)

            def show_points(coords, ax, marker_size=375):
                ax.scatter(coords[0], coords[1], color='green', marker='*', s=marker_size,
                           edgecolor='white', linewidth=1.25)

            def show_box(box, ax):
                x0, y0, w, h = box[0], box[1], box[2], box[3]
                ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

            for i, mask in enumerate(masks):
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                show_mask(mask['segmentation'], plt.gca())
                show_points(mask['point_coords'] * 1241 / 1024, plt.gca())
                show_box(mask['bbox'], plt.gca())
                plt.title(f"Mask {i + 1}, Score: {mask['predicted_iou']:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()
            """

            gt_label = np.ones((2, img.shape[0], img.shape[1]), dtype='uint16') * 65535
            for i in range(len(img_indices)):
                h, w = img_indices[i]
                gt_label[0, h, w] = img_label[i][0]
                gt_label[1, h, w] = instance_label[i][0]
            pseudo_label = gt_label.copy()
            for mask in reversed(masks):
                counter = Counter(gt_label[0][mask['segmentation']])
                counter[65535] = 0
                semantic_max = counter.most_common(1)[0][0]
                if semantic_max not in [0, 65535]:
                    pseudo_label[0][mask['segmentation']] = semantic_max
                    counter = Counter(gt_label[1][mask['segmentation']])
                    counter[65535] = 0
                    instance_max = counter.most_common(1)[0][0]
                    if instance_max not in [0, 65535]:
                        pseudo_label[1][mask['segmentation']] = instance_max

            pseudo_label[pseudo_label == 65535] = 0

            """
            gt_label[gt_label == 65535] = 0
            
            with open(config['dataset_params']['label_mapping'], 'r') as stream:
                import yaml
                semkittiyaml = yaml.safe_load(stream)
            color_map = semkittiyaml['color_map']
            learning_map_inv = semkittiyaml['learning_map_inv']
            color = []
            for i in learning_map_inv.values():
                color.append(color_map[i])
            color = np.array(color)
            color = np.fliplr(color)
            """

            """
            from PIL import Image
            tmp = Image.fromarray(img)
            tmp.save('img.png')
            tmp = Image.fromarray(color[gt_label[0]].astype('uint8'))
            tmp.save('semantic_gt_label.png')
            tmp = Image.fromarray(color[pseudo_label[0]].astype('uint8'))
            tmp.save('semantic_pseudo_label.png')
            tmp = Image.fromarray(gt_label[1].astype('uint8'))
            tmp.save('instance_gt_label.png')
            tmp = Image.fromarray(pseudo_label[1].astype('uint8'))
            tmp.save('instance_pseudo_label.png')
            """

            """
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(13, 4))
            plt.axis('off')
            plt.imshow(img)
            plt.show()
            
            plt.figure(figsize=(13, 4))
            plt.axis('off')
            plt.imshow(color[gt_label[0]])
            plt.show()
            
            plt.figure(figsize=(13, 4))
            plt.axis('off')
            plt.imshow(color[pseudo_label[0]])
            plt.show()
            
            plt.figure(figsize=(13, 4))
            plt.axis('off')
            plt.imshow(gt_label[1])
            plt.show()
            
            plt.figure(figsize=(13, 4))
            plt.axis('off')
            plt.imshow(pseudo_label[1])
            plt.show()
            """

            np.save(save_file, pseudo_label)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
