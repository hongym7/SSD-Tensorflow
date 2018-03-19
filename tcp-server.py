import socket
import sys
import traceback
import threading
from threading import Thread
import base64
import os
import math
import random
import time
import numpy as np
import tensorflow as tf
import cv2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
from struct import *

##GPU, SSD network, TF Session: Restore a checkpoint and keep a Session for all threads...##
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, device_count={'GPU': 2})
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = \
    ssd_vgg_preprocessing.preprocess_for_eval(img_input,
                                              None,
                                              None,
                                              net_shape,
                                              data_format,
                                              resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
ssd_time = time.time()
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
print('ssd_net_time: ', time.time() - ssd_time)
restore_time = time.time()
#TODO: ckpt_filename is from sys.argv
# Restore SSD model.
# ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = './checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
# ckpt_filename = './checkpoints/tfmodel/model.ckpt-77752'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
print('restore time: ', time.time() - restore_time)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


def main():
    start_server()


def start_server():
    host = "10.250.46.57"
    port = 8888         # arbitrary non-privileged port

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR flag tells the kernel to reuse a local socket in TIME_WAIT state, without waiting for its natural timeout to expire
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    print("Socket created")

    try:
        soc.bind((host, port))
    except:
        print("Bind failed. Error : " + str(sys.exc_info()))
        sys.exit()

    soc.listen(5)       # queue up to 5 requests
    print("Socket now listening")
    # infinite loop- do not reset for every requests
    while True:
        connection, address = soc.accept()
        ip, port = str(address[0]), str(address[1])
        print("Connected with " + ip + ":" + port)

        try:
            Thread(target=client_thread, args=(connection, ip, port), name=ip+':'+port).start()
        except:
            print("Thread did not start.")
            traceback.print_exc()

    soc.close()
    isess.close()


def receive_input(connection, max_buffer_size):
    client_input = b''

    # while True:
    #     client_input_part = connection.recv(max_buffer_size)
    #     client_input += client_input_part
    #     if len(client_input_part) < max_buffer_size:
    #         break

    # try:
    client_input = b''
    chunk = b''
    i = 0
    while True:
        chunk = connection.recv(max_buffer_size)
        if i == 0:
          packet_size = int.from_bytes(chunk[:4], byteorder='big', signed=False)
          i = 1
        # print('packet_size: ', packet_size, 'chuck size: ', len(chunk), 'length of client_input: ', len(client_input))

        if len(client_input) < packet_size:
            # Unreliable
            client_input += chunk
            # break
        if len(client_input) == packet_size:
            break
            # client_input += chunk
        print('packet_size: ', packet_size, 'chuck size: ', len(chunk), 'length of client_input: ', len(client_input))

    print('packet_size: ', packet_size, 'chuck size: ', len(chunk), 'length of client_input: ', len(client_input))
    client_input_size = sys.getsizeof(client_input)
    print('client_input_size: ', client_input)
    if client_input_size > max_buffer_size:
        print("The input size is greater than expected {}".format(client_input_size))

    # packet_size = int.from_bytes(client_input[:4], byteorder='big', signed=False)
    # image_size  = int.from_bytes(client_input[250:254], byteorder='big', signed=False)
    decoded_input = "aaaa"
    decoded_input += client_input[4:16].decode("utf8")  # decode and strip end of line
    decoded_input += client_input[16:50].decode("utf8")
    decoded_input += client_input[50:250].decode("utf8")
    decoded_input += "bbbb"
    decoded_input += str(client_input[254:])
    result = process_input(decoded_input)
    packet_size_image_size = [int.from_bytes(client_input[:4], byteorder='big', signed=False), int.from_bytes(client_input[250:254], byteorder='big', signed=False)]
    return result, packet_size_image_size, client_input


def process_input(input_str):
    print("Processing the input received from client")

    return str(input_str)


def calculateOriginalBox(x, y, w, h):
    top_left_x = x * w
    top_left_y = y * h

    return top_left_x, top_left_y

def client_thread(connection, ip, port, max_buffer_size = 3000000):
    is_active = True
    print('The current thread: ', threading.currentThread().getName())
    while is_active:
        client_input, sizes, rcv_data = receive_input(connection, max_buffer_size)
        print("client input length: ", len(client_input))

#################################################################################
        #parse the request
        packet_size = str(sizes[0])
        service_code = client_input[4:16]
        company_code = client_input[16:50]
        image_name = client_input[50:250]
        image_size = str(sizes[1])
        image_data = rcv_data[254:]

        print("client_input :\n", client_input)
        print("packet_size: ", packet_size)
        print("service_code: ", service_code)
        print("company_code: ", company_code)
        print("image_name: ", image_name)
        print("image_size: ", image_size)
        #TODO trim the string
###########################################################################################


        prediction_startime = time.time()
        # prediction_startime = time.time()
        # # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        # sessCreationTime = time.time()
        # isess = tf.InteractiveSession(config=config)
        # print('a session created: ', time.time() - sessCreationTime)
        # # Input placeholder.
        # net_shape = (300, 300)
        # data_format = 'NHWC'
        # img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # # Evaluation pre-processing: resize to SSD net shape.
        # image_pre, labels_pre, bboxes_pre, bbox_img = \
        #     ssd_vgg_preprocessing.preprocess_for_eval(img_input,
        #                                               None,
        #                                               None,
        #                                               net_shape,
        #                                               data_format,
        #                                               resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        # image_4d = tf.expand_dims(image_pre, 0)
        #
        # # Define the SSD model.
        # ssd_time = time.time()
        # reuse = True if 'ssd_net' in locals() else None
        # ssd_net = ssd_vgg_300.SSDNet()
        # with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        #     predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
        # print('ssd_net_time: ', time.time() - ssd_time)
        # restore_time= time.time()
        # # Restore SSD model.
        # # ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
        # ckpt_filename = './Checkpoint/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        # # ckpt_filename = './checkpoints/tfmodel/model.ckpt-77752'
        # isess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.restore(isess, ckpt_filename)
        # print('restore time: ', time.time() - restore_time)
        # # SSD default anchor boxes.
        # # ssd_anchors_time = time.time()
        # ssd_anchors = ssd_net.anchors(net_shape)
        # # print('ssd_anchor_time: ', time.time() - ssd_anchors_time)
        # Main image processing routine.
        def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
            # Run SSD network.
            rimg, rpredictions, rlocalisations, rbbox_img = \
                isess.run([image_4d, predictions, localisations, bbox_img],
                          feed_dict={img_input: img})

            # Get classes and bboxes from the net outputs.
            rclasses, rscores, rbboxes = \
                np_methods.ssd_bboxes_select(rpredictions,
                                             rlocalisations,
                                             ssd_anchors,
                                             select_threshold=select_threshold,
                                             img_shape=net_shape,
                                             num_classes=21,
                                             decode=True)

            rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
            rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
            rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
            return rclasses, rscores, rbboxes

        # Test on some demo image and visualize output.
        #path = './Test_Images/IMG_0522.jpg'


        #image_data1 is for some mockup.  image_data is for production
       # image_file_name = image_data.rstrip('\x00')
       # image_binary = base64.b64decode(image_file_name)

        #print('image: ', image_binary)
        # print(data[image_data])

        # bytearray(image_binary2)
        # k = open('a.jpg', 'wb')
        # k.write(image_binary2)
        # k.close()

        # bytearray(image_file_name)

        # with open('test.jpeg', 'wb') as f:
        #
        #     f.write(image_binary2) #image_binary
        #     im = cv2.imread('test.jpg')
        #     # height, width = im.shape[:2]

        start_time = time.time()
        with open('test.jpeg', 'wb') as f:
            f.write(image_data)  # image_binary
            im = cv2.imread('test.jpeg')
            height, width = im.shape[:2]


        print("Time for converting base64 to Image: {} seconds".format(time.time() - start_time))
        print('image width, height: (', width, ',', height, ')')
        # TODO: Can convert byte array into numpy array!!

        numpy_array_of_image = mpimg.imread(f.name)  # path)
        image_without_alpha = numpy_array_of_image[:, :, :3]

        rclasses, rscores, rbboxes = process_image(image_without_alpha)
        num_of_detected = rclasses.size

        print('object detected: ', num_of_detected)

        i=0

        class_cat = "{:<4}".format('1')
        response=''
        for rclass in rclasses:
            rclass_4 = "{:<4}".format(rclass)
            rscore = "{:8.3f}".format(rscores[i])
            rbbox0 = "{:8.3f}".format((rbboxes[i][1])*width)
            rbbox1 = "{:8.3f}".format((rbboxes[i][0])*height)
            print("Top Left Corner (x, y) = ({} , {})".format(rbbox0, rbbox1))
            #(ymin, xmin, ymax, xmax)
            rbbox_width = "{:8.3f}".format((rbboxes[i][3]-rbboxes[i][1])*width)
            rbbox_height = "{:8.3f}".format((rbboxes[i][2]-rbboxes[i][0])*height)
            print("width, height of the default box: ", rbbox_width, rbbox_height)
            response += class_cat + str(rclass_4) + rscore + rbbox0 + rbbox1 + rbbox_width + rbbox_height
            # print("object info [{}]: {}".format(i, response))
            i += 1

        print("object info's: {}".format(response))

        response1 = service_code + company_code + image_name
        response2 = "{:<4}".format(num_of_detected) + response

        print('packet_size: ', packet_size)
        print('service_code: ', service_code)
        print('company_code: ', company_code)
        print('image_name: ', image_name)
        print('image_size: ', image_size)

        print('rclasses: ', rclasses)
        print('rscores : ', rscores)
        print('rbbboxes : ', rbboxes)
        print('elapsed time: ', time.time() - prediction_startime)
        # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

###########################################################################################
        if "--QUIT--" in client_input:
            print("Client is requesting to quit")
            connection.close()
            print("Connection " + ip + ":" + port + " closed")
            is_active = False
        else:
            print("Processed request: {}".format(client_input))

            #send the response
            #packet_size(4):service_code(12):company_code(34):image_name(200):image_size(4):object_detected(4):object_info
            #client_input + detected_object + object_info's

            connection.sendall(rcv_data[0:4] + response1.encode("utf8") + rcv_data[250:254] + intToBytes(width) + intToBytes(height) +response2.encode("utf8"))


def intToBytes(n):
    b = bytearray([0, 0, 0, 0])  # init
    b[3] = n & 0xFF
    n >>= 8
    b[2] = n & 0xFF
    n >>= 8
    b[1] = n & 0xFF
    n >>= 8
    b[0] = n & 0xFF

    return b

if __name__ == "__main__":
    main()