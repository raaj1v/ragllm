# merge code with sumit's
import os
from PIL import Image, ImageDraw
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import torch
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import easyocr
from paddleocr import PaddleOCR

class ImageExtractor:
    def __init__(self, img_file_path, in_process_images_folder_path):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.reader = easyocr.Reader(['en'])
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.img_file_path = img_file_path
        self.in_process_images_folder_path = in_process_images_folder_path

    def remove_cropped_part(self, original_image_path, cropped_image_bbox, base_filename):
        original_img = Image.open(original_image_path).convert("RGBA")
        draw = ImageDraw.Draw(original_img)
        draw.rectangle(cropped_image_bbox, fill=(255, 255, 255, 0))
        result_img = original_img.convert("RGB")
        only_text_image_path = os.path.join(self.in_process_images_folder_path, f"{base_filename}_onlyTextImage.jpg")
        result_img.save(only_text_image_path)
        return result_img

    def plot_results(self, image, scores, labels, boxes, base_filename, original_image_path):
        for score, label, (xmin, ymin, xmax, ymax) in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
            if self.model.config.id2label[0] == 'table':
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                cropped_image_bbox = (xmin, ymin, xmax, ymax)
                self.remove_cropped_part(original_image_path, cropped_image_bbox, base_filename)
                detected_table_path = os.path.join(self.in_process_images_folder_path, f"{base_filename}_detectedTable.jpg")
                cropped_image.save(detected_table_path)
                break

    def detect_table(self, original_image_path, base_filename):
        image = Image.open(original_image_path).convert("RGB")
        width, height = image.size
        image.resize((int(width * 0.5), int(height * 0.5)))
        feature_extractor = DetrFeatureExtractor()
        encoding = feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding)
        
        results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
        self.plot_results(image, results['scores'], results['labels'], results['boxes'], base_filename, original_image_path)

    def table_detection_display(self, img_path):
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        dilated_image = self.thick_font(img_gray)
        _, img_bin = cv2.threshold(dilated_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = cv2.bitwise_not(img_bin)

        kernel_length_v = (np.array(img_gray).shape[1]) // 80
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
        im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
        vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

        kernel_length_h = (np.array(img_gray).shape[1]) // 20
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
        im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
        table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=1)
        _, table_segment = cv2.threshold(table_segment, 127, 255, cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        merged_contours = contours
        return self.process_contours(merged_contours, img)

    def thick_font(self, image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        return cv2.bitwise_not(image)

    def process_contours(self, merged_contours, img):
        full_list = []
        row = []
        first_iter = 0
        firsty = -1
        
        for c in merged_contours:
            x, y, w, h = cv2.boundingRect(c)
            if first_iter == 0:
                first_iter = 1
                firsty = y
            if firsty != y:
                row.reverse()
                full_list.append(row)
                row = []
            cropped = img[y:y + h, x:x + w]
            cell_content = self.extract_cell_content(cropped)
            row.append((cell_content, w))
            firsty = y
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if row:
            row.reverse()
            full_list.append(row)
        full_list.reverse()

        new_data = [[j[0] for j in i] for i in full_list]
        df = pd.DataFrame(new_data).applymap(lambda x: '' if pd.isna(x) else x)
        return tabulate(df, headers='firstrow', tablefmt='grid')

    def extract_cell_content(self, cropped):
        result = self.ocr.ocr(cropped, cls=True)
        if result and result[0] is not None and len(result[0]) > 0:
            bounds = [(line[0], line[1][0]) for line in result[0]]
            return "\n".join([bound[1] for bound in bounds])
        else:
            bounds = self.reader.readtext(cropped)
            return " ".join([bound[1] for bound in bounds])

    def extract_text(self, only_text_image_path):
        result = self.ocr.ocr(only_text_image_path, cls=True)
        if result is None:
            return "No Text Found in File"
        
        extracted_text = []
        for line in result:
            if line is None:
                continue
            for item in line:
                if item is None:
                    continue
                bbox, (text, score) = item
                extracted_text.append(text)
        
        return '\n'.join(extracted_text)

    def process_images(self):
        img_path = self.img_file_path
        base_filename = os.path.basename(img_path).split('.')[0]
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_scale_img_path = os.path.join(self.in_process_images_folder_path, f'{base_filename}_grey.jpg')
        cv2.imwrite(grey_scale_img_path, img_gray)
        original_image_path = grey_scale_img_path
        self.detect_table(original_image_path, base_filename)
        detected_table_path = os.path.join(self.in_process_images_folder_path, f"{base_filename}_detectedTable.jpg")
        only_text_image_path = os.path.join(self.in_process_images_folder_path, f"{base_filename}_onlyTextImage.jpg")
        extracted_data = "Extracted Table:\n"
        if os.path.exists(detected_table_path):
            extracted_data += self.table_detection_display(detected_table_path)
            os.remove(detected_table_path)
        extracted_data += "\nExtracted Text:\n"
        if os.path.exists(only_text_image_path):
            extracted_data += self.extract_text(only_text_image_path)
            os.remove(only_text_image_path)
        else:
            extracted_data += self.extract_text(original_image_path)
        os.remove(grey_scale_img_path)
        return extracted_data


