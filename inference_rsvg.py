import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import util.misc as utils
from util.misc import AverageMeter
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from datasets import build_dataset, get_coco_api_from_dataset
import opts
from torch.utils.data import DataLoader

from tools.colormap import colormap
import pandas as pd


colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

Visualize_bbox = True #False #True
save_visualize_path_prefix = "test"
version = "test"

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    utils.setup_for_distributed(args.rank == 0)

def main(args):
    args.masks = False
    
    if args.distributed:
        init_distributed_mode(args)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.visualize and utils.is_main_process():
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    test_dataset = build_dataset(args.dataset_file, image_set='test', args=args)
    
    if args.distributed:
        sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        sampler=sampler,
        pin_memory=True, 
        drop_last=False,
        num_workers=4
    )

    # model
    model, criterion, _ = build_model(args)
    device = torch.device(args.device)
    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.distributed:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    # start inference
    evaluate(test_loader, model, args)

def evaluate(test_loader, model, args):
    batch_time = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    device = args.device
    model.eval()
    end = time.time()

    count = 0
    results = []

    for batch_idx, (img, targets, dw, dh, img_path, ratio) in enumerate(test_loader):
        h_resize, w_resize = img.shape[-2:]
        img = img.to(device)
        captions = targets["caption"]
        size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
        target = {"size": size}

        with torch.no_grad():
            outputs = model(img, captions, [target])

        # single level selection
        # according to pred_logits, select the query index
        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid()  # [t, q, k]
        pred_score = pred_score.squeeze(0)  # [q, k]

        rescaled_bboxes = []
        for bbox in pred_bbox[0]:
            rescaled_bbox = rescale_bboxes(bbox.detach(), (w_resize, h_resize)).numpy()
            rescaled_bboxes.append(rescaled_bbox)
        rescaled_bboxes = [bbox.tolist() for bbox in rescaled_bboxes]

        # pred_scores = pred_scores.mean(0)  # [q, k]
        max_score, _ = pred_score.max(-1)  # [q,]
        _, max_ind = max_score.max(-1)  # [1,] # which query
        pred_bbox = pred_bbox[0, max_ind]  # [xc,yc, w_b, h_b]

        # xywh2xyxy
        pred_bbox = rescale_bboxes(pred_bbox.detach(), (w_resize, h_resize)).numpy()
        target_bbox = rescale_bboxes(targets["boxes"].squeeze(), (w_resize, h_resize)).numpy()

        # ratio = float(ratio)
        # x1, x2 = pred_bbox[0], pred_bbox[2]
        pred_bbox[0], pred_bbox[2] = (pred_bbox[0] - dw) / ratio, (pred_bbox[2] - dw) / ratio
        pred_bbox[1], pred_bbox[3] = (pred_bbox[1] - dh) / ratio, (pred_bbox[3] - dh) / ratio
        target_bbox[0], target_bbox[2] = (target_bbox[0] - dw) / ratio, (target_bbox[2] - dw) / ratio
        target_bbox[1], target_bbox[3] = (target_bbox[1] - dh) / ratio, (target_bbox[3] - dh) / ratio

        if Visualize_bbox and (not args.distributed or args.rank == 0):
            source_img = Image.open(img_path[0]).convert('RGB')  # PIL image

            draw = ImageDraw.Draw(source_img)
            draw_boxes = pred_bbox.tolist()

            # draw boxes
            xmin, ymin, xmax, ymax = draw_boxes[0:4]

            draw_boxes_gt = target_bbox.tolist()
            xmin_gt, ymin_gt, xmax_gt, ymax_gt = draw_boxes_gt[0:4]

            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[9]), width=2)
            draw.rectangle(((xmin_gt, ymin_gt), (xmax_gt, ymax_gt)), outline=(0, 255, 0), width=2)
            
            # save
            save_visualize_path_dir = os.path.join(save_visualize_path_prefix, version)
            if not os.path.exists(save_visualize_path_dir):
                os.makedirs(save_visualize_path_dir)
            img_name = img_path[0].split('/')[-1]

            count += 1
            img_name = f"{args.rank}_{count}_{img_name}" if args.distributed else f"{count}_{img_name}"

            save_visualize_path = os.path.join(save_visualize_path_dir, img_name)
            source_img.save(save_visualize_path)

        # box iou
        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox)

        pred_bbox_list = pred_bbox.tolist()
        result_item = {
            'iou': iou.item(),
            'image_path': img_path[0],
            'captions': captions,
            'pred_bbox': [pred_bbox_list],
            'pred_score': [[1.0]]
        }
        results.append(result_item)

        cumInterArea = np.sum(np.array(interArea.data.numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.numpy()))
        # accuracy
        accu5 = np.sum(np.array((iou.data.numpy() > 0.5), dtype=float)) / 1
        accu6 = np.sum(np.array((iou.data.numpy() > 0.6), dtype=float)) / 1
        accu7 = np.sum(np.array((iou.data.numpy() > 0.7), dtype=float)) / 1
        accu8 = np.sum(np.array((iou.data.numpy() > 0.8), dtype=float)) / 1
        accu9 = np.sum(np.array((iou.data.numpy() > 0.9), dtype=float)) / 1

        # metrics  7
        meanIoU.update(torch.mean(iou).item(), img.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)

        acc5.update(accu5, img.size(0))
        acc6.update(accu6, img.size(0))
        acc7.update(accu7, img.size(0))
        acc8.update(accu8, img.size(0))
        acc9.update(accu9, img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0 and (not args.distributed or args.rank == 0):
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.avg:.3f}\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t' \
                        'acc@0.7: {acc7.avg:.4f}\t' \
                        'acc@0.8: {acc8.avg:.4f}\t' \
                        'acc@0.9: {acc9.avg:.4f}\t' \
                        'meanIoU: {meanIoU.avg:.4f}\t' \
                        'cumuIoU: {cumuIoU:.4f}\t' \
                .format( \
                batch_idx, len(test_loader), batch_time=batch_time, \
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
            print(print_str)

    if args.distributed:
        all_metrics = torch.tensor([acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg, 
                                    meanIoU.avg, inter_area.sum, union_area.sum]).cuda()
        dist.all_reduce(all_metrics)
        all_metrics /= args.world_size
        
        all_results = [None for _ in range(args.world_size)]
        dist.all_gather_object(all_results, results)
        
        if args.rank == 0:
            merged_results = []
            for res_list in all_results:
                merged_results.extend(res_list)
            results = merged_results
            
            acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg, meanIoU.avg = all_metrics[:6]
            inter_area.sum, union_area.sum = all_metrics[6:8]
    
    if not args.distributed or args.rank == 0:
        final_str = 'acc@0.5: {acc5.avg:.4f}\t' 'acc@0.6: {acc6.avg:.4f}\t' 'acc@0.7: {acc7.avg:.4f}\t' \
                    'acc@0.8: {acc8.avg:.4f}\t' 'acc@0.9: {acc9.avg:.4f}\t' \
                    'meanIoU: {meanIoU.avg:.4f}\t' 'cumuIoU: {cumuIoU:.4f}\t' \
            .format(acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                    meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
        print(final_str)
        print(version)

        results_df = pd.DataFrame(results)
        results_df.to_csv('data/inference_results.csv', index=False)

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(box1[0]), torch.tensor(box1[1]), torch.tensor(box1[2]), torch.tensor(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(box2[0]), torch.tensor(box2[1]), torch.tensor(box2[2]), torch.tensor(box2[3])

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return (inter_area + 1e-6) / (union_area + 1e-6), inter_area, union_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=0)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')  # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Refer_RSVG inference script', parents=[opts.get_args_parser()])
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='distributed rank')
    args = parser.parse_args()
    main(args)