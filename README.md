# SignSense

## Folder Structure

`client/` contains the FE clients (Desktop Python script and Android)

`tools/` contains all the Python scripts for training, processing, and predictions

`server/` contains the multi-threaded Python server that will run the prediction logic

`data/` contains the generated numpy data for training

`videos/` contains the raw recorded ASL videos

`models/` contains the trained models for the prediction script

## Running Locally

## live predict with GPU (working on linux and Windows)


To use GPU with Tensorflow, install CUDA 11 and cuDNN 8 onto your system.
If you don't have them already, delete your current CUDA installation and follow the steps at https://gist.github.com/kmhofmann/cee7c0053da8cc09d62d74a6a4c1c5e4. 
Make sure you download version 460 of the driver. You'll have to create a Nvidia account to download cuDNN.

On Windows, you may need to add the cuDNN folder to your PATH and [change a couple of dll names](https://stackoverflow.com/questions/65608713/tensorflow-gpu-could-not-load-dynamic-library-cusolver64-10-dll-dlerror-cuso).
