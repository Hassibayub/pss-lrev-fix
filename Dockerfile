FROM tensorflow/tensorflow:latest-gpu


RUN apt update && apt upgrade -y
RUN pip install --upgrade pip 

COPY requirements.txt .
RUN pip install -r requirements.txt 


COPY *.py . 
COPY *.ipynb .


