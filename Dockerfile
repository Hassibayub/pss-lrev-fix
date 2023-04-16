FROM tensorflow/tensorflow:latest-gpu


RUN ["apt", "update"] 
RUN ["apt", "upgrade", "-y"]

RUN ["pip", "install", "--upgrade", "pip"]
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


WORKDIR /src

COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"] 


COPY . . 



