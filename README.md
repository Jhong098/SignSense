# SignSense

Real-time American Sign Language (ASL) interpreter using MediaPipe, Tensorflow, and Keras.

SignSense can recognize up to 21 signs with a latency of less than 2 seconds.

![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/16534325/113086932-7d1bdf80-9197-11eb-9e52-1df0fe744688.gif)

- [PPT Presentation](https://www.youtube.com/watch?v=68mIaMoLarQ)
- [Quick Demo Video](https://www.youtube.com/watch?v=LgDpJ2IbYMM)

## Folder Structure

`client.py` contains the Python client that runs MediePipe on the webcam feed and displays predictions

`server.py` contains the Python server that will run the prediction logic

`tools/` contains all the Python scripts for training, processing, predictions, and misc. utils

`data/` contains the generated numpy data for training

`models/` contains the trained models

## Running Locally

Two options
1. Run `client.py` and `server.py`
2. Run `tools/live_predict.py`

## Running with GPU (working on linux and Windows)

To use GPU with Tensorflow, install CUDA 11 and cuDNN 8 onto your system.
If you don't have them already, delete your current CUDA installation and follow the steps at https://gist.github.com/kmhofmann/cee7c0053da8cc09d62d74a6a4c1c5e4. 
Make sure you download version 460 of the driver. You'll have to create a Nvidia account to download cuDNN.

On Windows, you may need to add the cuDNN folder to your PATH and [change a couple of dll names](https://stackoverflow.com/questions/65608713/tensorflow-gpu-could-not-load-dynamic-library-cusolver64-10-dll-dlerror-cuso).

## Contributing Data
Instructions can be found [here](data_gen_inst.md)
