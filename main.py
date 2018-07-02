#!/usr/bin/python3
# -*- coding: utf-8 -*-

#
# Description: given an input image of a user, replace the face of a selected actor(actress)
#              in a video.
# Args:
#      video address: the directory where your video is stored
#      user's image: the image uploaded by the user
#      actor(actress)'s image: the actor's image stored in the database
#      temp_address: the directory where the temporary files are generated
#      output_video: the directory where the output video is generated
#

import subprocess as sp
import faceswap
import time
import cv2
import sys
import os
import glob
import shutil
import face_recognition
import numpy

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
    cmd = 'ffmpeg -hide_banner -loglevel panic -i ' + video_address + ' -qmin 1 -q:v 1 -vf fps=' + \
        fps + ' ' + images_address + 'out%04d.jpg'
    sp.call(cmd, shell=True)
    return images_address


def convert_images2videos(swap_images_address, output_video):
    '''
        Description: Convert images into videos based on its FPS
        Params: images address after face swap
        Ouput: video address
    '''

    cmd = 'ffmpeg -hide_banner -loglevel panic -i ' + swap_images_address + \
        'output%4d.jpg -r ' + fps + ' ' + output_video
    sp.call(cmd, shell=True)


if __name__ == "__main__":

    start = time.time()
    video_address = sys.argv[1]
    user_image = sys.argv[2]
    actor_images = sys.argv[3]
    temp_address = sys.argv[4]
    output_video = sys.argv[5]
    # step 1. convert a video into images
    images_address = convert_videos2images(video_address, output_video_address)

    actor_face = face_recognition.load_image_file(actor_images)
    actor_encoding = face_recognition.face_encodings(actor_face)[0]
    swap_images_addr = ''
    if (temp_address.endswith('/')):
        swap_images_addr = temp_address + "merged_images/"
    else:
        swap_images_addr = temp_address + "/merged_images/"

    if not os.path.exists(swap_images_addr):
        os.mkdir(swap_images_addr)
    try:
        im2, landmarks2 = faceswap.read_im_and_landmarks_user(user_image)
    except faceswap.NoFaces:
        print("No face in the uploaded user image")
        sys.exit()
    except faceswap.TooManyFaces:
        print("Too many faces in the uploaded user image")
        sys.exit()
    # except TooManyFaces:
    #    sys.exit('%d', -2)

    # mask = faceswap.get_face_mask(im2, landmarks2)

    actors = [{'img_addr': file, 'output_addr': swap_images_addr, 'idx': file[file.index(".jpg") - 4: file.index(
        ".jpg")], "encoding": actor_encoding, "im2": im2, "landmark": landmarks2} for idx, file in enumerate(glob.glob(images_address + "*.jpg"))]

    # step 2. swap faces, it should then return
    faceswap.FaceSwap(actors)

    # step 3. merge images into a video
    convert_images2videos(swap_images_addr, output_video)

    # delete all temparary iamges in iamges_address and swap_images
    shutil.rmtree(images_address, ignore_errors=True)
    shutil.rmtree(swap_images_addr, ignore_errors=True)
    print("All done", time.time() - start)
