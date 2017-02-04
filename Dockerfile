FROM nvidia/cuda:8.0-cudnn5-devel

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
ENV no_proxy *.local,169.254/16
ENV NO_PROXY *.local,169.254/16

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy \
		swig \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
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
	python -m ipykernel.kernelspec


# Install Tensorflow
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

# Install Keras
RUN pip --no-cache-dir install git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# additional libraries
RUN pip --no-cache-dir install --upgrade gensim

# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]