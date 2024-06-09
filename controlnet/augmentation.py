import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import v2

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input cropped images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/cropped_images') # <YOUR_OWN_PATH>
    parser.add_argument('--output-path', type=str, help='path of output corrupted images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/corrupted_images') # <YOUR_OWN_PATH>
    args = parser.parse_args()
    
    random.seed(7729)
    transforms = {0: v2.RandomResizedCrop(size=(256, 256), scale=(0.08, 1.0), ratio=(1.0, 1.0)), 1: v2.RandomRotation(degrees=30, fill=(255, 255, 255)), \
                  2: v2.GaussianBlur(kernel_size=(11, 11), sigma=(1.0, 10.0))} 
    test_indices = random.sample(list(range(1, 21)), 5)
    
    for i in range(len(os.listdir(args.input_path))):
        folder = os.listdir(args.input_path)[i]
        os.makedirs(os.path.join(args.output_path, folder.replace("CROPPED", "CORRUPTED")), exist_ok=True)
        num_images = len(os.listdir(os.path.join(args.input_path, folder)))
        # approximately equal amount of each type of transformation within one folder
        transform_indices = random.choices(list(transforms.keys()), weights=[1/len(transforms)] * len(transforms), k=num_images)
        for j in range(num_images):
            crop_image = os.listdir(os.path.join(args.input_path, folder))[j]
            # print("crop_image", crop_image)
            # print("transform", transform_indices[j])
            im = Image.open(os.path.join(args.input_path, folder, crop_image)).convert("RGB")
            im_np = np.asarray(im)
            im_tensor = torch.tensor(im_np, dtype=torch.uint8).permute(2, 0, 1)
            # apply augmentation
            transform_ind = transform_indices[j]
            transform = transforms[transform_ind]
            if transform_ind==0:
                transformed_tensor = transform(im_tensor)
                im_np = transformed_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                im_pil = Image.fromarray(im_np)
            else:
                im_pil = transform(im).resize((256, 256))
            im_pil.save(os.path.join(args.output_path, folder.replace("CROPPED", "CORRUPTED"),crop_image))                
                
                
            