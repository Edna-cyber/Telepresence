import os
import shutil
import json

def move_files(src_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    counter = 0
    # Walk through the source directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            new_filename = str(counter)+".png"
            dest_path = os.path.join(dest_dir, new_filename)
            # Move file to destination directory
            try: 
                shutil.copy(file_path, dest_path)
            except:
                print("File already exists.")
            counter += 1

move_files("/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/corrupted_images", "/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/source")
move_files("/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/groundtruth_images", "/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/target")

data = []

for i in range(len(os.listdir("/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/source"))):
    entry = {
        "source": "source/{}.png".format(str(i)),
        "target": "target/{}.png".format(str(i)),
        "prompt": "Recover a clean and high resolution image for me"
    }
    data.append(entry)

with open('/usr/project/xtmp/rz95/Telepresence/controlnet/training/self_portrait/prompt.json', 'a') as json_file: # 'w'
    json.dump(data, json_file, indent=4)

