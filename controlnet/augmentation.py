import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import v2

class SpeckleNoiseTransform(v2.Transform):
    def _transform(self, inpt, params: dict):
        noise = torch.randn(inpt.size())
        noisy = inpt + inpt * noise
        return torch.clamp(noisy, 0, 255)

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input cropped images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/cropped_images') # <YOUR_OWN_PATH>
    parser.add_argument('--gt-path', type=str, help='path of ground truth images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/groundtruth_images') # <YOUR_OWN_PATH>
    parser.add_argument('--corrupted-path', type=str, help='path of corrupted images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/corrupted_images') # <YOUR_OWN_PATH>
    args = parser.parse_args()
    
    random.seed(7729)
    augmentations = {0: v2.RandomResizedCrop(size=(256, 256), scale=(0.08, 1.0), ratio=(1.0, 1.0)), 1: v2.RandomRotation(degrees=30, fill=(255, 255, 255)), 2: None}
    transforms = {0: v2.GaussianBlur(kernel_size=(11, 11), sigma=(1.0, 10.0)), 1: SpeckleNoiseTransform()}

    test_indices = random.sample(list(range(1, 21)), 5)
    
    for i in range(len(os.listdir(args.input_path))):
        folder = os.listdir(args.input_path)[i]
        os.makedirs(os.path.join(args.gt_path, folder.replace("CROPPED", "GROUND_TRUTH")), exist_ok=True)
        os.makedirs(os.path.join(args.corrupted_path, folder.replace("CROPPED", "CORRUPTED")), exist_ok=True)
        num_images = len(os.listdir(os.path.join(args.input_path, folder)))
        # approximately equal amount of each type of transformation within one folder
        augmentation_indices = random.choices(list(augmentations.keys()), weights=[1/len(augmentations)] * len(augmentations), k=num_images)
        transform_indices = random.choices(list(transforms.keys()), weights=[1/len(transforms)] * len(transforms), k=num_images)
        for j in range(num_images):
            crop_image = os.listdir(os.path.join(args.input_path, folder))[j]
            # print("crop_image", crop_image)
            # print("augmentation", augmentation_indices[j])
            # print("corruption", transform_indices[j])
            im = Image.open(os.path.join(args.input_path, folder, crop_image)).convert("RGB")
            im_np = np.asarray(im)
            im_tensor = torch.tensor(im_np, dtype=torch.uint8).permute(2, 0, 1)
            
            # apply augmentation and blurring / adding noise
            augmentation_ind = augmentation_indices[j]
            augmentation = augmentations[augmentation_ind]
            transform_ind = transform_indices[j]
            transform = transforms[transform_ind]
            
            if augmentation_ind==0:
                augmented_tensor = augmentation(im_tensor)
                im_np = augmented_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                gt = Image.fromarray(im_np)
            else:
                if augmentation_ind==1:
                    gt = augmentation(im).resize((256, 256))
                elif augmentation_ind==2:
                    gt = im.resize((256, 256))
            if transform_ind==0:
                corrupted = transform(gt)
            else:
                gt_tensor = torch.tensor(np.asarray(gt), dtype=torch.uint8).permute(2, 0, 1)
                corrupted_tensor = transform(gt_tensor)
                corrupted_np = corrupted_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                corrupted = Image.fromarray(corrupted_np)
            gt.save(os.path.join(args.gt_path, folder.replace("CROPPED", "GROUND_TRUTH"), crop_image))
            corrupted.save(os.path.join(args.corrupted_path, folder.replace("CROPPED", "CORRUPTED"), crop_image))    
           
                
                
            