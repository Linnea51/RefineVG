import os
import csv
import ast
import numpy as np
import random
from PIL import Image
import xml.etree.ElementTree as ET
import shutil
csv_file_path = "data/inference_results.csv"
output_crops_path = "data/DIOR_RSVG_croppedtest/JPEGImages"
bbox_output_file = "data/DIOR_RSVG_croppedtest/expanded_bboxes.txt"

os.makedirs(output_crops_path, exist_ok=True)

bbox_file = open(bbox_output_file, 'w')

with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            image_path = row['image_path']
            image_name = os.path.basename(image_path)
            
            pred_bbox = ast.literal_eval(row['pred_bbox'])
            pred_score = ast.literal_eval(row['pred_score'])
            
            pred_bbox = np.array(pred_bbox)
            pred_score = np.array(pred_score).flatten()
            
            if len(pred_score) > 0:
                best_idx = np.argmax(pred_score)
                bbox = pred_bbox[best_idx]
                
                img = Image.open(image_path)
                
                x1, y1, x2, y2 = map(int, bbox)
                w, h = x2 - x1, y2 - y1
                
                expand_ratio = 0.25
                
                expand_left = int(w * expand_ratio)
                expand_top = int(h * expand_ratio)
                expand_right = int(w * expand_ratio)
                expand_bottom = int(h * expand_ratio)
                
                img_w, img_h = img.size
                ex1 = max(0, x1 - expand_left)
                ey1 = max(0, y1 - expand_top)
                ex2 = min(img_w, x2 + expand_right)
                ey2 = min(img_h, y2 + expand_bottom)
                
                cropped_img = img.crop((ex1, ey1, ex2, ey2))
                
                resized_img = cropped_img.resize((480, 480), Image.LANCZOS)
                
                output_image_path = os.path.join(output_crops_path, image_name)
                resized_img.save(output_image_path)
                
                bbox_file.write(f"{image_name}: ({ex1}, {ey1}), ({ex2}, {ey2})\n")
                
                print(f"Process: {image_name}")
                
        except Exception as e:
            print(f"Process {image_name} error: {e}")

bbox_file.close()

print(f"Save as {output_crops_path}")
print(f"Save as {bbox_output_file}")

anno_src_dir = "data/DIOR_RSVG_addpatch/Annotations"
anno_dst_dir = "data/DIOR_RSVG_croppedtest/Annotations"
os.makedirs(anno_dst_dir, exist_ok=True)

for xml_file in os.listdir(anno_src_dir):
    if xml_file.endswith(".xml"):
        src_path = os.path.join(anno_src_dir, xml_file)
        dst_path = os.path.join(anno_dst_dir, xml_file)
        tree = ET.parse(src_path)
        root = tree.getroot()
        size = root.find("size")
        if size is not None:
            width = size.find("width")
            height = size.find("height")
            if width is not None:
                width.text = "480"
            if height is not None:
                height.text = "480"
        else:
            print(f"Warning: No <size> tag in {src_path}")

        obj = root.find("object")
        if obj is not None:
            name_elem = obj.find("name")
            desc_elem = obj.find("description")
            if name_elem is not None and desc_elem is not None:
                desc_elem.text = name_elem.text

        tree.write(dst_path)
print("All annotation XMLs saved to data/DIOR_RSVG_croppedtest/Annotations with width/height=480 and description set to name.")


for split in ["test.txt"]:
    src = os.path.join("data/DIOR_RSVG_addpatch", split)
    dst = os.path.join("data/DIOR_RSVG_croppedtest", split)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    else:
        print(f"{src} does not exist, skipping.")


src = "data/inference_results.csv"
dst = "data/inference_results_fullimage.csv"
if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"Copied {src} to {dst}")
else:
    print(f"{src} does not exist, skipping.")