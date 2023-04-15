FROM tensorflow/tensorflow:latest-gpu


RUN ["apt", "update"] 
RUN ["apt", "upgrade", "-y"]

RUN ["pip", "install", "--upgrade", "pip"] 


WORKDIR /src

COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"] 


COPY . . 



