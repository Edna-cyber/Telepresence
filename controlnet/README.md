# Telepresence

- [x] Converted self-portrait videos (MOV/MP4) into a sequence JPG images [3522 images total]
- [x] Separated the person from background using MODNet 
- [x] Cropped out images within face-centered bounding boxes using Mediapipe
- [x] Created corrupted/augmented images using Gaussian Blur, Random Rotation, and Random Crop
```
conda create -n tele python=3.9
conda activate tele
pip install -r requirements.txt
```
Run in this order: <br>
```python3 MODNet_inference.py``` => ```python3 face_detector.py``` => ```python3 augmentation.py```
