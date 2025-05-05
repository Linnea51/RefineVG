import os
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

restored_bboxes_path = "data/DIOR_RSVG_croppedtest/restored_bboxes.txt"
csv_file_path = "data/inference_results_fullimage.csv"
annotations_path = "data/DIOR_RSVG_addpatch/Annotations"
output_path = "data/fusion.txt"

bbox_dict1 = {}
with open(restored_bboxes_path, 'r') as f:
    for line in f:
        parts = line.strip().split(': ')
        image_name = parts[0]
        coords = parts[1].split('), (')
        left_bottom = tuple(map(float, coords[0][1:].split(', ')))
        right_top = tuple(map(float, coords[1][:-1].split(', ')))
        bbox_dict1[image_name] = (left_bottom, right_top)

df = pd.read_csv(csv_file_path)
bbox_dict2 = {}
for _, row in df.iterrows():
    image_name = os.path.basename(row['image_path'])
    pred_bboxes = eval(row['pred_bbox'])
    pred_scores = eval(row['pred_score'])
    
    max_score_index = pred_scores.index(max(pred_scores))
    best_bbox = pred_bboxes[max_score_index]
    bbox_dict2[image_name] = ((best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]))

fused_bboxes = {}
for image_name in set(bbox_dict1.keys()) & set(bbox_dict2.keys()):
    bbox1 = bbox_dict1[image_name]
    bbox2 = bbox_dict2[image_name]
    
    fused_bbox = (
        ((bbox1[0][0] + bbox2[0][0])/2, (bbox1[0][1] + bbox2[0][1])/2),
        ((bbox1[1][0] + bbox2[1][0])/2, (bbox1[1][1] + bbox2[1][1])/2)
    )
    fused_bboxes[image_name] = fused_bbox

def calculate_iou(box1, box2):
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[1][0], box2[1][0])
    y2 = min(box1[1][1], box2[1][1])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])
    area2 = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])
    union = area1 + area2 - intersection
    
    return intersection, union, intersection / union if union > 0 else 0

class_ious = defaultdict(list)

class_precisions = defaultdict(lambda: {"pre0.1": 0, "pre0.5": 0, "pre0.6": 0, "pre0.7": 0, "pre0.8": 0, "pre0.9": 0, "count": 0})

total_intersection = 0
total_union = 0

with open(output_path, 'w') as f:
    for image_name, fused_bbox in fused_bboxes.items():
        f.write(f"{image_name}: ({fused_bbox[0][0]:.4f}, {fused_bbox[0][1]:.4f}), ({fused_bbox[1][0]:.4f}, {fused_bbox[1][1]:.4f})\n")
        
        xml_path = os.path.join(annotations_path, os.path.splitext(image_name)[0] + '.xml')
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                gt_bbox = (
                    (float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)),
                    (float(bndbox.find('xmax').text), float(bndbox.find('ymax').text))
                )
                
                intersection, union, iou = calculate_iou(fused_bbox, gt_bbox)
                class_ious[class_name].append(iou)
                
                total_intersection += intersection
                total_union += union

                class_precisions[class_name]["count"] += 1
                
                if iou >= 0.1:
                    class_precisions[class_name]["pre0.1"] += 1
                if iou >= 0.5:
                    class_precisions[class_name]["pre0.5"] += 1
                if iou >= 0.6:
                    class_precisions[class_name]["pre0.6"] += 1
                if iou >= 0.7:
                    class_precisions[class_name]["pre0.7"] += 1
                if iou >= 0.8:
                    class_precisions[class_name]["pre0.8"] += 1
                if iou >= 0.9:
                    class_precisions[class_name]["pre0.9"] += 1

cum_iou = total_intersection / total_union if total_union > 0 else 0

print("\nAverage IoU per class:")
for class_name, ious in class_ious.items():
    mean_iou = sum(ious) / len(ious)
    print(f"{class_name}: {mean_iou:.4f} (samples: {len(ious)})")

all_ious = [iou for ious in class_ious.values() for iou in ious]
miou = sum(all_ious) / len(all_ious)
print(f"\nOverall mIoU: {miou:.4f}")
print(f"Overall cumIoU: {cum_iou:.4f}")

print("\nPrecision@0.1, 0.5, 0.6, 0.7, 0.8, 0.9 per class:")
for class_name, stats in class_precisions.items():
    pre01 = stats["pre0.1"] / stats["count"] if stats["count"] > 0 else 0
    pre05 = stats["pre0.5"] / stats["count"] if stats["count"] > 0 else 0
    pre06 = stats["pre0.6"] / stats["count"] if stats["count"] > 0 else 0
    pre07 = stats["pre0.7"] / stats["count"] if stats["count"] > 0 else 0
    pre08 = stats["pre0.8"] / stats["count"] if stats["count"] > 0 else 0
    pre09 = stats["pre0.9"] / stats["count"] if stats["count"] > 0 else 0
    print(f"{class_name}: pre0.1 = {pre01:.4f}, pre0.5 = {pre05:.4f}, pre0.6 = {pre06:.4f}, pre0.7 = {pre07:.4f}, pre0.8 = {pre08:.4f}, pre0.9 = {pre09:.4f}")

total_pre01 = sum(stats["pre0.1"] for stats in class_precisions.values()) / sum(stats["count"] for stats in class_precisions.values())
total_pre05 = sum(stats["pre0.5"] for stats in class_precisions.values()) / sum(stats["count"] for stats in class_precisions.values())
total_pre06 = sum(stats["pre0.6"] for stats in class_precisions.values()) / sum(stats["count"] for stats in class_precisions.values())
total_pre07 = sum(stats["pre0.7"] for stats in class_precisions.values()) / sum(stats["count"] for stats in class_precisions.values())
total_pre08 = sum(stats["pre0.8"] for stats in class_precisions.values()) / sum(stats["count"] for stats in class_precisions.values())
total_pre09 = sum(stats["pre0.9"] for stats in class_precisions.values()) / sum(stats["count"] for stats in class_precisions.values())
print(f"\nOverall precision@0.1: {total_pre01:.4f}")
print(f"Overall precision@0.5: {total_pre05:.4f}")
print(f"Overall precision@0.6: {total_pre06:.4f}")
print(f"Overall precision@0.7: {total_pre07:.4f}")
print(f"Overall precision@0.8: {total_pre08:.4f}")
print(f"Overall precision@0.9: {total_pre09:.4f}")

print(f"\nFusion results have been saved to: {output_path}")