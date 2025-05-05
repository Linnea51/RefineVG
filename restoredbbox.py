import os
import pandas as pd

expanded_bboxes_path = "data/DIOR_RSVG_croppedtest/expanded_bboxes.txt"
inference_results_path = "data/inference_results.csv"
output_bboxes_path = "data/DIOR_RSVG_croppedtest/restored_bboxes.txt"

expanded_bboxes = {}
with open(expanded_bboxes_path, 'r') as f:
    for line in f:
        parts = line.strip().split(': ')
        image_name = parts[0]
        coords = parts[1].split('), (')
        left_bottom = tuple(map(int, coords[0][1:].split(', ')))
        right_top = tuple(map(int, coords[1][:-1].split(', ')))
        expanded_bboxes[image_name] = (left_bottom, right_top)

df = pd.read_csv(inference_results_path)

restored_bboxes = []

for index, row in df.iterrows():
    image_name = os.path.basename(row['image_path']) 
    pred_bboxes = eval(row['pred_bbox'])
    pred_scores = eval(row['pred_score'])

    max_score_index = pred_scores.index(max(pred_scores))
    best_bbox = pred_bboxes[max_score_index]

    if image_name in expanded_bboxes:
        left_bottom, right_top = expanded_bboxes[image_name]

        width = right_top[0] - left_bottom[0]
        height = right_top[1] - left_bottom[1]

        best_bbox[0] = best_bbox[0] * width / 480
        best_bbox[1] = best_bbox[1] * height / 480
        best_bbox[2] = best_bbox[2] * width / 480
        best_bbox[3] = best_bbox[3] * height / 480

        restored_xmin = best_bbox[0] + left_bottom[0]
        restored_ymin = best_bbox[1] + left_bottom[1]
        restored_xmax = restored_xmin + (best_bbox[2] - best_bbox[0])
        restored_ymax = restored_ymin + (best_bbox[3] - best_bbox[1])

        restored_bboxes.append(f"{image_name}: ({restored_xmin}, {restored_ymin}), ({restored_xmax}, {restored_ymax})")

with open(output_bboxes_path, 'w') as f:
    for bbox in restored_bboxes:
        f.write(bbox + '\n')

print(f"bbox saved at {output_bboxes_path}。")