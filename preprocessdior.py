import os
import xml.etree.ElementTree as ET
import shutil
from xml.dom import minidom
from PIL import Image
import random

class DataProcessor:
    def __init__(self):
        # Input paths
        self.input_anno_path = "data/DIOR_RSVG/Annotations"
        self.input_images_path = "data/DIOR_RSVG/JPEGImages"
        
        # Output paths
        self.output_anno_path = "data/DIOR_RSVG_addpatch/Annotations"
        self.output_images_path = "data/DIOR_RSVG_addpatch/JPEGImages"
        
        # Create output directories
        os.makedirs(self.output_anno_path, exist_ok=True)
        os.makedirs(self.output_images_path, exist_ok=True)
        os.makedirs("data/DIOR_RSVG_addpatch", exist_ok=True)  # 创建主文件夹
        
        self.count = 0
        self.original_count = 0  # To track number of original samples

    def process_single_object(self, anno_file_path, original_image_path):
        """Process each xml file"""
        root = ET.parse(anno_file_path).getroot()
        img = Image.open(original_image_path)
        img_width, img_height = img.size
        
        for member in root.findall('object'):
            # 首先验证是否有 bndbox
            bndbox = member.find('bndbox')
            if bndbox is None:
                print(f"Warning: No bounding box found in {anno_file_path}")
                continue

            # Step 1: Save original split version
            new_root = ET.Element("annotation")
            filename_element = ET.SubElement(new_root, "filename")
            filename_element.text = f"{self.count:05d}.jpg"
            
            # Copy basic information
            source_element = ET.SubElement(new_root, "source")
            database_element = ET.SubElement(source_element, "database")
            database_element.text = "DIOR"
            
            # Copy size information
            size_element = ET.SubElement(new_root, "size")
            width_element = ET.SubElement(size_element, "width")
            width_element.text = str(img_width)
            height_element = ET.SubElement(size_element, "height")
            height_element.text = str(img_height)
            depth_element = ET.SubElement(size_element, "depth")
            depth_element.text = "3"
            
            # Copy object information with fixed order
            object_element = ET.SubElement(new_root, "object")
            
            # 1. name element (first)
            name_element = ET.SubElement(object_element, "name")
            name_element.text = member.find('name').text
            
            # 2. pose element (second)
            pose_element = ET.SubElement(object_element, "pose")
            pose_element.text = "Unspecified"
            
            # 3. bndbox element (third)
            bndbox_element = ET.SubElement(object_element, "bndbox")
            for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                coord_elem = ET.SubElement(bndbox_element, coord)
                coord_elem.text = member.find(f'bndbox/{coord}').text
            
            # 4. description element (fourth) - 保持原始description
            description_element = ET.SubElement(object_element, "description")
            original_description = member.find('description')
            if original_description is not None:
                description_element.text = original_description.text
            else:
                description_element.text = member.find('name').text  # 只有在没有description时才使用name
            
            # Save split XML and copy original image
            xml_str = ET.tostring(new_root, encoding='utf-8')
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
            
            with open(os.path.join(self.output_anno_path, f"{self.count:05d}.xml"), "w") as f:
                f.write(pretty_xml)
            
            shutil.copy(original_image_path, 
                       os.path.join(self.output_images_path, f"{self.count:05d}.jpg"))
            
            self.count += 1
            
        self.original_count = self.count  # Save the count of original samples

    def process_crop_resize(self, anno_file_path, original_image_path):
        """Process cropped and resized version"""
        root = ET.parse(anno_file_path).getroot()
        img = Image.open(original_image_path)
        img_width, img_height = img.size
        
        for member in root.findall('object'):
            try:
                # Get original bbox
                xmin = int(member.find('bndbox/xmin').text)
                ymin = int(member.find('bndbox/ymin').text)
                xmax = int(member.find('bndbox/xmax').text)
                ymax = int(member.find('bndbox/ymax').text)
                
                # Validate bbox coordinates
                if xmin >= xmax or ymin >= ymax:
                    print(f"Invalid bbox in {anno_file_path}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
                    continue
                
                # Expand bounding box
                width = xmax - xmin
                height = ymax - ymin
                
                expand_ratio_ymin = random.uniform(0, 0.5)
                expand_ratio_ymax = random.uniform(0, 0.5)
                expand_ratio_xmin = random.uniform(0, 0.5)
                expand_ratio_xmax = random.uniform(0, 0.5)
                
                new_xmin = max(0, int(xmin - width * expand_ratio_xmin))
                new_xmax = min(img_width, int(xmax + width * expand_ratio_xmax))
                new_ymin = max(0, int(ymin - height * expand_ratio_ymin))
                new_ymax = min(img_height, int(ymax + height * expand_ratio_ymax))
                
                # Additional validation after expansion
                if new_xmin >= new_xmax or new_ymin >= new_ymax:
                    print(f"Invalid expanded bbox in {anno_file_path}")
                    continue
                
                # Crop and resize
                cropped_img = img.crop((new_xmin, new_ymin, new_xmax, new_ymax))
                resized_img = cropped_img.resize((480, 480), Image.LANCZOS)
                
                # Create XML for cropped and resized version
                final_root = ET.Element("annotation")
                filename_element = ET.SubElement(final_root, "filename")
                filename_element.text = f"{self.count:05d}.jpg"
                
                source_element = ET.SubElement(final_root, "source")
                database_element = ET.SubElement(source_element, "database")
                database_element.text = "DIOR"
                
                size_element = ET.SubElement(final_root, "size")
                width_element = ET.SubElement(size_element, "width")
                width_element.text = "480"
                height_element = ET.SubElement(size_element, "height")
                height_element.text = "480"
                depth_element = ET.SubElement(size_element, "depth")
                depth_element.text = "3"
                
                # Calculate new coordinates
                crop_width = new_xmax - new_xmin
                crop_height = new_ymax - new_ymin
                
                scaled_xmin = int((xmin - new_xmin) * (480 / crop_width))
                scaled_ymin = int((ymin - new_ymin) * (480 / crop_height))
                scaled_xmax = int((xmax - new_xmin) * (480 / crop_width))
                scaled_ymax = int((ymax - new_ymin) * (480 / crop_height))
                
                object_element = ET.SubElement(final_root, "object")
                name_element = ET.SubElement(object_element, "name")
                name_element.text = member.find('name').text
                pose_element = ET.SubElement(object_element, "pose")
                pose_element.text = "Unspecified"
                
                bndbox_element = ET.SubElement(object_element, "bndbox")
                xmin_element = ET.SubElement(bndbox_element, "xmin")
                xmin_element.text = str(scaled_xmin)
                ymin_element = ET.SubElement(bndbox_element, "ymin")
                ymin_element.text = str(scaled_ymin)
                xmax_element = ET.SubElement(bndbox_element, "xmax")
                xmax_element.text = str(scaled_xmax)
                ymax_element = ET.SubElement(bndbox_element, "ymax")
                ymax_element.text = str(scaled_ymax)
                
                description_element = ET.SubElement(object_element, "description")
                description_element.text = member.find('name').text  # 使用name作为description
                
                # Save final version
                xml_str = ET.tostring(final_root, encoding='utf-8')
                pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
                
                with open(os.path.join(self.output_anno_path, f"{self.count:05d}.xml"), "w") as f:
                    f.write(pretty_xml)
                
                resized_img.save(os.path.join(self.output_images_path, f"{self.count:05d}.jpg"))
                
                self.count += 1
                
            except Exception as e:
                print(f"Error processing {anno_file_path}: {str(e)}")
                continue

    def process_all(self):
        """Process all files"""
        def filelist(root, file_type):
            return [os.path.join(directory_path, f) 
                   for directory_path, directory_name, files in os.walk(root) 
                   for f in files if f.endswith(file_type)]
        
        annotations = filelist(self.input_anno_path, '.xml')
        
        # First pass: create split versions
        print("Processing original split versions...")
        for anno_file_path in annotations:
            root = ET.parse(anno_file_path).getroot()
            image_file_name = root.find("./filename").text
            original_image_path = os.path.join(self.input_images_path, image_file_name)
            
            if os.path.exists(original_image_path):
                self.process_single_object(anno_file_path, original_image_path)
            else:
                print(f"Image file not found: {original_image_path}")
        
        print(f"Completed original split versions. Generated {self.original_count} samples")
        
        # Second pass: create cropped and resized versions
        print("Processing cropped and resized versions...")
        for anno_file_path in annotations:
            root = ET.parse(anno_file_path).getroot()
            image_file_name = root.find("./filename").text
            original_image_path = os.path.join(self.input_images_path, image_file_name)
            
            if os.path.exists(original_image_path):
                self.process_crop_resize(anno_file_path, original_image_path)
            else:
                print(f"Image file not found: {original_image_path}")
        
        print(f"Processing completed!")
        print(f"Original split samples: {self.original_count}")
        print(f"Total samples: {self.count}")
        print(f"All results saved in {self.output_anno_path} and {self.output_images_path}")
        
        # Update training file - 在新文件夹下创建train.txt
        if os.path.exists("data/DIOR_RSVG/train.txt"):
            # 读取原始train.txt
            with open("data/DIOR_RSVG/train.txt", "r") as f:
                original_indices = [int(line.strip()) for line in f.readlines()]
            
            # 在新文件夹下创建train.txt
            with open("data/DIOR_RSVG_addpatch/train.txt", "w") as f:
                # 写入原始索引
                for idx in original_indices:
                    f.write(f"{idx}\n")
                # 写入新索引（原始索引 + original_count）
                for idx in original_indices:
                    f.write(f"{idx + self.original_count}\n")
            print(f"New training file created in data/DIOR_RSVG_addpatch/train.txt with {len(original_indices) * 2} samples")

        # 复制 val.txt 和 test.txt 到新目录
        for split in ["val.txt", "test.txt"]:
            src = os.path.join("data/DIOR_RSVG", split)
            dst = os.path.join("data/DIOR_RSVG_addpatch", split)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"{src} does not exist, skipping.")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()

