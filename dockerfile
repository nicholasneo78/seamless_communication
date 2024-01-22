FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y gcc build-essential libsndfile1 git sox wget ffmpeg sudo python3-pip vim \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
RUN rm -rf $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/ruamel*
ADD requirements.txt .
RUN pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# # clone the seamless-m4t repo
# RUN git clone https://github.com/facebookresearch/seamless_communication.git
# RUN cd seamless_communication && pip install .

# build from source
ADD . .
RUN pip install .

# declare port used by jupyterlab
EXPOSE 8888

# set default command for jupyterlab
CMD ["jupyter" ,"lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]

#docker container starts with bash
WORKDIR /seamless_communication

RUN ["bash"]