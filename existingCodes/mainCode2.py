# merge code with sumit's
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from image_Extraction_lib_2 import ImageExtractor

img_file_path = r'/data/QAAPI/75054933.jpeg'
in_process_images_folder_path = r'/data/QAAPI/inProcessImages'

extractor = ImageExtractor(img_file_path, in_process_images_folder_path)
data = extractor.process_images()

print("--------- start -------")
print(data)
print("------- end ----------")