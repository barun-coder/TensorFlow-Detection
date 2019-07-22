#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

input_video = "./input_images_and_videos/The Dancing Traffic Light Manikin by smart.mp4"
input_video ="http://107.85.197.18:8080/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000"
input_video = "./input_images_and_videos/Ti_Way_one.mp4"
# input_video="http://180.48.75.229/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000" # shop entry
# input_video="http://219.162.167.54:80/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000"
# input_video="http://166.149.142.29:8000/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000"
# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')

#object_counting_api.object_counting(input_video, detection_graph, category_index, 0) # for counting all the objects, disabled color prediction

#object_counting_api.object_counting(input_video, detection_graph, category_index, 1) # for counting all the objects, enabled color prediction


# targeted_objects = "person" # (for counting targeted objects) change it with your targeted objects
# fps = 24 # change it with your input video fps
# width = 1280    # change it with your input video width
# height = 960 # change it with your input vide height
# is_color_recognition_enabled = 0
#
# #object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
#
# object_counting_api.object_counting(0, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects


targeted_objects = "person,car,bicycle"
fps = 24 # change it with your input video fps
width = 1920 # change it with your input video width
height = 1080 # change it with your input vide height
is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
roi = 200 # roi line position
deviation = 3 # the constant that represents the object counting area


# object_counting_api.cumulative_object_counting_x_axis("http://107.85.197.18:8080/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000", detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation) # counting all the objects
object_counting_api.custom_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
# object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
