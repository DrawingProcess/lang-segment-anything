# pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

from  PIL  import  Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image

import cv2
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage

import os

base_path = '/data/csj000714/data/nerf_custom/'
input = 'headset/'
output = 'headset_only/'
input_path = base_path + input
output_path = base_path + output

model = LangSAM()
text_prompt = 'headset'

img_list = []
folder_list = os.listdir(input_path)
for folder in folder_list:
    if os.path.isdir(input_path + folder):
        file_list = os.listdir(input_path + folder)
        img_list.extend([input_path + folder + '/' + file for file in file_list if file.endswith(".png")])
print(img_list)

for img in img_list:
    image_pil = Image.open(img).convert("RGB")
    masks, boxes, labels, logits = model.predict(image_pil, text_prompt)

    tf_toTensor = ToTensor() 
    image = tf_toTensor(image_pil) * masks

    tf_toPILImage = ToPILImage() 
    image = tf_toPILImage(image)

    output_img_path = img.replace(input, output)
    image = image.save(output_img_path)
    print ("Success Save Img: ", output_img_path)

# image = draw_image(image_pil, masks, boxes, labels)
# cv2.imwrite('frame_00001.png', image)