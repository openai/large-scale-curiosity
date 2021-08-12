FROM tensorflow/tensorflow:1.5.1-devel-gpu-py3
RUN apt update && apt upgrade -y

# Add newer python version
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN python3 -V


RUN apt install -y cmake libopenmpi-dev python3.6-dev zlib1g-dev python3-pip
RUN apt install -y ffmpeg libsm6 libxext6 python3.6-tk
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==1.5.1
RUN pip3 install opencv-python gym mpi4py pillow scikit-image
RUN pip3 install gym gym[atari] tensorboardX
RUN pip3 install opencv-python
RUN git clone https://github.com/openai/baselines.git
WORKDIR baselines
RUN git checkout 7c520852d9cf4eaaad326a3d548efc915dc60c10
RUN pip3 install -e .
RUN git clone https://github.com/openai/large-scale-curiosity /root/large-scale-curiosity
RUN apt install unrar
WORKDIR /root
RUN wget -O roms.rar http://www.atarimania.com/roms/Roms.rar
RUN unrar e roms.rar
RUN unzip ROMS -d ROMS
RUN python3 -m atari_py.import_roms ROMS
WORKDIR /root/large-scale-curiosity
