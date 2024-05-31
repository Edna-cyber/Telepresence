#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import subprocess

# Function to run a shell command
def run_shell_command(command):
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.decode())

# run_shell_command("pip install mediapipe")
# run_shell_command("wget -q -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")

# Visualization utilities
from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Cropped image within the bounding boxes.
  """
  cropped_image = image.copy() # annotated_image
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = (bbox.origin_x, bbox.origin_y)
    end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height) # bbox.width=bbox.height=543
    # Manually enlarge bounding_box
    center_point = ((start_point[0]+end_point[0])/2, (start_point[1]+end_point[1])/2)
    start_point = (int(center_point[0]-700), int(center_point[1]-700))
    end_point = (int(center_point[0]+700), int(center_point[1]+700))
    cropped_image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    # cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3) 
    # Draw keypoints
    # for keypoint in detection.keypoints:
    #   keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
    #                                                  width, height)
    #   color, thickness, radius = (0, 255, 0), 2, 2
    #   cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # # Draw label and score
    # category = detection.categories[0]
    # category_name = category.category_name
    # category_name = '' if category_name is None else category_name
    # probability = round(category.score, 2)
    # result_text = category_name + ' (' + str(probability) + ')'
    # text_location = (MARGIN + bbox.origin_x,
    #                  MARGIN + ROW_SIZE + bbox.origin_y)
    # cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #             FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return cropped_image 

# Running inference and visualizing the results
# The final step is to run face detection on your selected image. This involves creating your FaceDetector object, loading your image, running detection, and finally, the optional step of displaying the image with visualizations.
# You can check out the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/face_detector/python) to learn more about configuration options that this solution supports.

# Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input matte images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/matte_images') # <YOUR_OWN_PATH>
    parser.add_argument('--intermediate-path', type=str, help='path of matte images with white margin', default='/usr/project/xtmp/rz95/Telepresence/controlnet/white_margin_images') # <YOUR_OWN_PATH>
    parser.add_argument('--output-path', type=str, help='path of output cropped images', default='/usr/project/xtmp/rz95/Telepresence/controlnet/cropped_images') # <YOUR_OWN_PATH>
    args = parser.parse_args()
    i = 0 ###
    for folder in os.listdir(args.input_path):
        if i==1:
            break
        os.makedirs(os.path.join(args.intermediate_path, folder.replace("FOREGROUND_MATTE", "WHITE_MARGIN")), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, folder.replace("FOREGROUND_MATTE", "CROPPED")), exist_ok=True)
        i += 1
        for matte_image in os.listdir(os.path.join(args.input_path, folder)):
            IMAGE_FILE = os.path.join(args.input_path, folder, matte_image)
            image = mpimg.imread(IMAGE_FILE) # 1080 * 1920
            # STEP 1: Add white margin to the input image. 
            white_image = np.ones((3000, 3000, 3), dtype=np.uint8) * 255
            mediapipe_image_np = np.array(image) * 255
            x_offset = (3000 - 1080) // 2
            y_offset = (3000 - 1920) // 2
            white_image[x_offset:x_offset+1080, y_offset:y_offset+1920, :] = mediapipe_image_np
            plt.imsave(os.path.join(args.intermediate_path, folder.replace("FOREGROUND_MATTE", "WHITE_MARGIN"), matte_image), white_image, format='png')
            
            # STEP 2: Load the input image.
            WHITE_IMAGE_FILE = os.path.join(args.intermediate_path, folder.replace("FOREGROUND_MATTE", "WHITE_MARGIN"), matte_image)
            image = mp.Image.create_from_file(WHITE_IMAGE_FILE) 
            
            # STEP 3: Detect faces in the input image.
            detection_result = detector.detect(image)
            # STEP 4: Process the detection result. 
            image_copy = np.copy(image.numpy_view())
            crop_image = visualize(image_copy, detection_result)    
            plt.imsave(os.path.join(args.output_path, folder.replace("FOREGROUND_MATTE", "CROPPED"), matte_image), crop_image, format='png')
            