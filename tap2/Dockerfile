# start from a CUDA with devel (needed for apex)
FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

# install apex
RUN apt-get update && apt-get install -y git && apt-get clean
RUN git clone https://github.com/nvidia/apex /apex && \
  cd /apex && \
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

### Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8

# only care about python3
RUN ln -s $(which python3.6) /usr/bin/python
# RUN ln -s $(which pip3) /usr/bin/pip

RUN mkdir /app

COPY ai-c-complex-e2e-qa /app/ai-c-complex-e2e-qa

RUN cd /app/ai-c-complex-e2e-qa && pip install --editable .
