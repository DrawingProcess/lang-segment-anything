# pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

from  PIL  import  Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image

import cv2
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import to_pil_image

import os

base_path = '/data/csj000714/data/nerf_custom/'
input = 'headset/'
input_path = base_path + input

output_mask = 'headset_mask/'
output = 'headset_alphabg/'

model = LangSAM()
text_prompt = 'umbrella'

img_list = []
folder_list = os.listdir(input_path)
for folder in folder_list:
    if os.path.isdir(input_path + folder):
        file_list = os.listdir(input_path + folder)
        img_list.extend([input_path + folder + '/' + file for file in file_list if file.endswith(".png")])
# img_list.sort()
print(img_list)

for img in img_list:
    image_pil = Image.open(img).convert('RGB')
    masks, boxes, labels, logits = model.predict(image_pil, text_prompt)
    # print(image_pil.size)

    # tf_toPILImage = ToPILImage() 
    # masks = to_pil_image(masks)
    if len(boxes) > 0:
        masks = Image.fromarray((255*masks[0]).numpy().astype(np.uint8))
    else:
        print("masks empty(not detected): ", img)
        masks = Image.new('L', (image_pil.size[0], image_pil.size[1]))
    output_mask_path = img.replace(input, output_mask)
    masks.save(output_mask_path)

    # print(masks.size)

    im_rgba = image_pil.copy()
    im_rgba.putalpha(masks)
    # print(im_rgba.size)

    output_img_path = img.replace(input, output)
    im_rgba = im_rgba.save(output_img_path)
    print ("Success Save Img: ", output_img_path)

# image = draw_image(image_pil, masks, boxes, labels)
# cv2.imwrite('frame_00001.png', image)