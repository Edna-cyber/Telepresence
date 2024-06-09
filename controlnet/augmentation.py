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
    transforms = {0: v2.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0), ratio=(1.0, 1.0)), 1: v2.ScaleJitter(target_size=(256,256), scale_range=(0.5, 1.5)), 2: v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))}
    test_indices = random.sample(list(range(1, 21)), 5)
    
    for i in range(len(os.listdir(args.input_path))):
        folder = os.listdir(args.input_path)[i]
        os.makedirs(os.path.join(args.output_path, folder.replace("CROPPED", "CORRUPTED")), exist_ok=True)
        num_images = len(os.listdir(folder))
        transform_indices = random.choices(list(transforms.keys()), weights=[1/len(transforms)] * len(transforms), k=num_images)
        for j in range(num_images):
            crop_image = os.listdir(folder)[j]
            im = Image.open(os.path.join(args.input_path, folder, crop_image))
            im = np.asarray(im) * 255
            im_tensor = torch.tensor(im, dtype=torch.uint8)
            if i in test_indices:
                transform = transforms[transform_indices[j]]
                transformed_tensor = transform(im_tensor)
                im_np = im_tensor.numpy()
                im_pil = Image.fromarray(im_np)
                im_pil.save('output_image.png') ####
                
                
                
            