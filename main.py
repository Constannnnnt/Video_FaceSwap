#!/usr/bin/python3
# -*- coding: utf-8 -*-

#
# Description: given an input image of a user, replace the face of a selected actor(actress)
#              in a video.
# Args:
#      video address: the directory where your video is stored
#      user's image: the image uploaded by the user
#      actor(actress)'s image: the actor's image stored in the database
#      output_video_address: the directory where the output video is generated
#

import subprocess as sp
import faceswap
import time
import cv2
import sys
import os
import shutil

# global variable
fps = 0


def convert_videos2images(video_address, output_video_address):
    '''
        Description: Convert videos into images based on its FPS
        Params: video adress
        Ouput: image address
    '''
    video = cv2.VideoCapture(video_address)
    global fps
    (major_ver, _, _) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    fps = str(int(fps))
    images_address = ''
    if (output_video_address.endswith('/')):
        images_address = output_video_address + 'generated_images/'
    else:
        images_address = output_video_address + '/generated_images/'
    if not os.path.exists(images_address):
        os.makedirs(images_address)
    cmd = 'ffmpeg -i ' + video_address + ' -qmin 1 -q:v 1 -vf fps=' + \
        fps + ' ' + images_address + 'out%04d.jpg'
    sp.call(cmd, shell=True)
    return images_address


def convert_images2videos(swap_images_address, output_video_address):
    '''
        Description: Convert images into videos based on its FPS
        Params: images address after face swap
        Ouput: video address
    '''
    if (not output_video_address.endswith('/')):
        output_video_address = output_video_address + '/'
    cmd = 'ffmpeg -i ' + swap_images_address + \
        'output%d.jpg -r ' + fps + ' ' + output_video_address + 'output.mp4'
    sp.call(cmd, shell=True)


if __name__ == "__main__":
    start = time.clock()

    video_address = sys.argv[1]
    users_image = sys.argv[2]
    actor_images = sys.argv[3]
    output_video_address = sys.argv[4]

    # step 1. convert a video into images
    images_address = convert_videos2images(video_address, output_video_address)

    # step 2. swap faces, it should then return
    swap_images_address = faceswap.FaceSwap(
        users_image, actor_images, images_address, output_video_address)

    print(swap_images_address)

    # step 3. merge images into a video
    convert_images2videos(swap_images_address, output_video_address)

    # delete all temparary iamges in iamges_address and swap_images
    for the_file in os.listdir(images_address):
        file_path = os.path.join(images_address, the_file)
        try:
            if (os.path.isfile(file_path) and the_file.endswith(".jpg")):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    os.rmdir(images_address)

    for the_file in os.listdir(swap_images_address):
        file_path = os.path.join(swap_images_address, the_file)
        try:
            if (os.path.isfile(file_path) and the_file.endswith(".jpg")):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    os.rmdir(swap_images_address)
    print('All done', time.clock() - start)
