# Video FaceSwap

Given a video clip, a user profile and an actor(actress) profile in the video, Swap the actor's face with the user's face. In a word, just replace the face of an actor with the input image for fun.

## Prerequisite

Install the prerequisites before using this module

1. Install *Cmake*
2. Install GCC(version>=4.8)
3. Install ffmpeg
4. ```shell
    pip3.5 install -r requirements.txt
    ```

**Note**: You will also need the *facial landmark detector*. The trained model can be downloaded from the [source](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2). Then, you can place it in the same directory where `main.py` is placed.

## Usage

```python3
python main.py /video_address /user_face_address /selected_actor_address /output_video_address
```

## Issue

1. To accelerate the program, *multiprocessing* is used, but it will comsume all your memory. If you are not interested in the computation time, you can comment this part out in `faceswap.py` and use sequential procedures.

    ```python
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool.map(faceswapper, actors)
    pool.close()
    pool.join()
    ```

## Credits

Thanks to [face_recognition](https://github.com/ageitgey/face_recognition) by [Adam Geitgey](https://github.com/ageitgey) and [faceswap](https://github.com/matthewearl/faceswap) by [Matthew Earl](https://github.com/matthewearl).
