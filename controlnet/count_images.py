import os

total = 0
for file in os.listdir("/usr/project/xtmp/rz95/Telepresence/controlnet/raw_images"): #<YOUR_OWN_PATH>
    underscore = file.rindex("_")
    total += int(file[underscore+1:])
print("There are {} images total.".format(total)) #3522


    
    