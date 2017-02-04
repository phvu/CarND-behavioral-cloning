FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

MAINTAINER Vu Pham <nobdy@nogroup.com>

ARG TENSORFLOW_VERSION=0.12.1
ARG KERAS_VERSION=1.2.1

# setup proxy for user
ENV PROXY_SERVER http://proxy:8080
ENV http_proxy $PROXY_SERVER
ENV https_proxy $PROXY_SERVER
ENV ftp_proxy $PROXY_SERVER
ENV all_proxy $PROXY_SERVER
ENV HTTP_PROXY $PROXY_SERVER
ENV HTTPS_PROXY $PROXY_SERVER
ENV FTP_PROXY $PROXY_SERVER
ENV ALL_PROXY $PROXY_SERVER
ENV no_proxy *.local,169.254/16,localhost,127.0.0.1
ENV NO_PROXY *.local,169.254/16,localhost,127.0.0.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libopenblas-dev \
	liblapack-dev \
        pkg-config \
	python \
        python3-dev \
        python3-pip \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
	tcl-dev \
	tk-dev \	
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

RUN python --version
RUN python3 --version

# Add SNI support to Python
#RUN pip3 --no-cache-dir install \
#		pyopenssl \
#		ndg-httpsclient \
#		pyasn1

RUN pip3 install --upgrade pip

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python3-numpy \
		python3-scipy \
		python3-nose \
		python3-h5py \
		python3-skimage \
		python3-matplotlib \
		python3-pandas \
		python3-sklearn \
		python3-sympy \
		swig \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install --upgrade ipython && \
	pip3 --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python3 -m ipykernel.kernelspec


# Install Tensorflow
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

# Install Keras
RUN pip3 --no-cache-dir install git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# additional libraries
RUN pip3 --no-cache-dir install --upgrade gensim

# Make sure CUDNN is detected
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/libcudnn.so.5 /usr/local/cuda/lib64/libcudnn.so

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]
