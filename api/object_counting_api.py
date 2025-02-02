#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------
import datetime
from datetime import date

import tensorflow as tf
import csv
import cv2
import numpy as np

from Database import database
from utils import visualization_utils as vis_util

# Variables
total_passed_vehicle = 0  # using it to count vehicles

def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation):
        total_passed_vehicle = 0
       
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])


        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes),
                                                                                                             np.squeeze(classes).astype(np.int32),
                                                                                                             np.squeeze(scores),
                                                                                                             category_index,
                                                                                                             x_reference = roi,
                                                                                                             deviation = deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)
                               
                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                  cv2.line(input_frame, (roi,0), (roi, height), (0, 0xFF, 0), 5)
                else:
                  cv2.line(input_frame, (roi,0), (roi, height), (0xFF, 0,0), 5)

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Pedestrians: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )


                cv2.putText(
                    input_frame,
                    'HS Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                #

                #

                output_movie.write(input_frame)
                print ("writing frame")
                cv2.imshow('object counting',input_frame)
                '''  if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])'''
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

def cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation):
        total_passed_vehicle = 0        

        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

               # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             2,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes),
                                                                                                             np.squeeze(classes).astype(np.int32),
                                                                                                             np.squeeze(scores),
                                                                                                             category_index,
                                                                                                             y_reference = roi,
                                                                                                             deviation = deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:                  
                  cv2.line(input_frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
                else:
                  cv2.line(input_frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)
                
                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )               
                
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )

                output_movie.write(input_frame)
                print ("writing frame")
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if (csv_line != "not_available"):
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        size, direction = csv_line.split(',')
                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(counting_mode) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)


                output_movie.write(input_frame)
                print ("writing frame")
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()


def custom_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):
    # initialize .csv
    with open('object_counting_report.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

    # input video
    cap = cv2.VideoCapture(input_video)
    objectType=""
    total_vehicle = 0
    total_pedesteriane: int = 0
    total_others: int = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    height = 0
    width = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                # counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                #                                                                                       input_frame,
                #                                                                                       1,
                #                                                                                       is_color_recognition_enabled,
                #                                                                                       np.squeeze(boxes),
                #                                                                                       np.squeeze(classes).astype(np.int32),
                #                                                                                       np.squeeze(scores),
                #                                                                                       category_index,
                #                                                                                       # x_reference=200,
                #                                                                                       y_reference = 200,
                #                                                                                       deviation=3,
                #                                                                                       use_normalized_coordinates=True,
                #                                                                                       line_thickness=4)

                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             2,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
                                                                                                             y_reference=200,
                                                                                                             deviation=3,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)


                if (len(counting_mode) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0, 255, 255), 2,
                                cv2.FONT_HERSHEY_SIMPLEX)
                #
                    # splitct= counting_mode.split(',')
                    for x in counting_mode.split(','):
                        print(x)
                        counts = 0
                        objectType = x
                        objectsplit=objectType.split(':')
                        counts= int(objectsplit[objectsplit.__len__()-1])

                        if (objectType.__contains__('car')):
                            total_vehicle = total_vehicle + counts
                        elif (objectType.__contains__('person')):
                            total_pedesteriane = total_pedesteriane + counts
                        else:
                            total_others= total_others + counts
                    # size=len(splitct)
                              #



                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicle: ' + str(total_vehicle),
                    (10, 105),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'Detected Person: ' + str(total_pedesteriane),
                    (10, 70),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'X Line',
                    (545, 200 - 10),
                    font,
                    0.6,
                    (0xFF,0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    input_frame,
                    'Y Line',
                    (545, 200 - 10),
                    font,
                    0.6,
                    (0, 0xFF ,0),
                    2,
                    cv2.LINE_AA,
                )
                output_movie.write(input_frame)
                print("writing frame")
                cv2.imshow('object counting', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if (csv_line != "not_available"):
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        size, direction = csv_line.split(',')
                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()


def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object, fps, width, height):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        the_result = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                cv2.imshow('object counting',input_frame)

                output_movie.write(input_frame)
                print ("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()

def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):     
        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            

       

        input_frame = cv2.imread(input_video)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # insert information text to video frame
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Visualization of the results of a detection.        
        counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                                                                                              1,
                                                                                              is_color_recognition_enabled,
                                                                                              np.squeeze(boxes),
                                                                                              np.squeeze(classes).astype(np.int32),
                                                                                              np.squeeze(scores),
                                                                                              category_index,
                                                                                              use_normalized_coordinates=True,
                                                                                              line_thickness=4)
        if(len(counting_mode) == 0):
            cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
        else:
            cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
        
        cv2.imshow('tensorflow_object counting_api',input_frame)        
        cv2.waitKey(0)

        return counting_mode

def writeinexcel(objecttype,counter):
    today = datetime.now()
    # dd/mm/YY
    d1 = today.strftime("%d/%m/%Y")
    t1 = today.strftime("%H:%M:%S")
    print(today)
    print(d1)
    print(t1)
    with open('custom.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line =d1, t1, objecttype, counter
        writer.writerows([csv_line.split(',')])

def cumulative_object_custom_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps,
                                      width, height, roi, deviation):
    total_passed_vehicle = 0

    # initialize .csv
    with open('custom.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Date, Time, Object, Count"
        writer.writerows([csv_line.split(',')])


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

    # input video
    cap = cv2.VideoCapture(input_video)

    total_car = 0
    total_pedesteriane: int = 0
    total_bicycle: int = 0
    total_truck: int = 0
    total_motorcycle: int = 0
    total_others: int = 0

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()
                print(ret)
                if not ret:
                    print("end of the video file...")
                    break
                else:
                    input_frame = frame
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # insert information text to video frame
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Visualization of the results of a detection.
                    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                                 input_frame,
                                                                                                                 1,
                                                                                                                 is_color_recognition_enabled,
                                                                                                                 np.squeeze(
                                                                                                                     boxes),
                                                                                                                 np.squeeze(
                                                                                                                     classes).astype(
                                                                                                                     np.int32),
                                                                                                                 np.squeeze(
                                                                                                                     scores),
                                                                                                                 category_index,
                                                                                                                 x_reference=roi,
                                                                                                                 deviation=deviation,
                                                                                                                 use_normalized_coordinates=True,
                                                                                                                 line_thickness=4)

                    # when the vehicle passed over line and counted, make the color of ROI line green
                    print("Values:" + str(counter)+" : "+str(csv_line)+" : "+str(counting_mode))
                    if counter == 1:
                        cv2.line(input_frame, (roi, 0), (roi, height), (0, 0xFF, 0), 2)
                        if (csv_line.__contains__('car')):
                            total_car = total_car + counter
                            database.mysqlInsert("Car",counter)
                        elif (csv_line.__contains__('person')):
                            total_pedesteriane = total_pedesteriane + counter
                            database.mysqlInsert("Person", counter)
                        elif (csv_line.__contains__('bicycle')):
                            total_bicycle = total_bicycle + counter
                            database.mysqlInsert("Bicycle", counter)
                        elif (csv_line.__contains__('truck')):
                            total_truck = total_truck + counter
                            database.mysqlInsert("Truck", counter)
                        elif (csv_line.__contains__('motorcycle')):
                            total_motorcycle = total_motorcycle + counter
                            database.mysqlInsert("MoterCycle", counter)
                        else:
                            total_others = total_others + counter
                            database.mysqlInsert("Other", counter)
                    else:
                        cv2.line(input_frame, (roi, 0), (roi,height), (0xFF, 0, 0), 2)



                    total_passed_vehicle = total_passed_vehicle + counter

                    # insert information text to video frame
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        'All: ' + str(total_passed_vehicle),
                        (10, 30),
                        font,
                        0.6,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        'M0otorCycle: ' + str(total_motorcycle),
                        (10, 130),
                        font,
                        0.6,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        'Truck: ' + str(total_truck),
                        (10, 110),
                        font,
                        0.6,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        'bicycle: ' + str(total_bicycle),
                        (10, 90),
                        font,
                        0.6,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        'Car: ' + str(total_car),
                        (10, 70),
                        font,
                        0.6,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )

                    cv2.putText(
                        input_frame,
                        'Person: ' + str(total_pedesteriane),
                        (10, 50),
                        font,
                        0.6,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                print("restart  video file...")
                output_movie.write(input_frame)
                print("writing frame")
                cv2.imshow('object counting', input_frame)
                '''  if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])'''
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

