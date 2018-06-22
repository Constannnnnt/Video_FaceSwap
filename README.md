# Video FaceSwap

Given a video clip, a user profile and an actor(actress) profile in the video, Swap the actor's face with the user's face. In a word, just replace the face of an actor with the input image for fun.

## Prerequisite

Install the prerequisites before using this module

**Mac/Linux**

```shell
brew install ffmepg
pip3.5 install -r requirements.txt
```

**On Windows**

1. install the ffmepg by downloading the windows build (zip file) and add it into the path.

2. ```shell
    pip3.5 install -r requirements.txt
    ```

**Note**: You will also need the *facial landmark detector*. The trained model can be downloaded from the [source](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2).

## Usage

```python3
python main.py /video_address /user_face_address /selected_actor_address /output_video_address
```

## Credits

Thanks to [face_recognition](https://github.com/ageitgey/face_recognition) by [Adam Geitgey](https://github.com/ageitgey) and [faceswap](https://github.com/matthewearl/faceswap) by [Matthew Earl](https://github.com/matthewearl).
