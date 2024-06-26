import json
import cv2
import numpy as np

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/prompt.json', 'rt') as f: # <YOUR_OWN_PATH>
            file_content = f.read()
            self.data = json.loads(file_content)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/' + source_filename) # <YOUR_OWN_PATH>
        target = cv2.imread('/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/' + target_filename) # <YOUR_OWN_PATH>

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

dataset = MyDataset()
print(len(dataset)) # 3503

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt) # “Recover a clean and high resolution image for me”
print(jpg.shape) # (256, 256, 3)
print(hint.shape) # (256, 256, 3)