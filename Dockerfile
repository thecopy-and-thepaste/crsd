FROM tensorflow/tensorflow:2.2.1-gpu-py3-jupyter

RUN apt-get update -y && \
    apt-get install -y git wget vim && \
    apt-get install libgl1-mesa-dev -y && \
    apt-get install poppler-utils -y

WORKDIR /lib

ADD . . 

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV BASE_DIR /lib/crsd

ENV PYTHONPATH=$PYTHONPATH:$BASE_DIR:$BASE_DIR/crsd
WORKDIR ${BASE_DIR}

CMD [ "echo", "Image built" ]
