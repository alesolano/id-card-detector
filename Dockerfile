FROM python:3.7

COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc ffmpeg libsm6 libxext6
RUN pip install -r requirements.txt