
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 && \
        apt-get update && \
        apt-get install libnvinfer5=5.0.2-1+cuda10.0

RUN file="$(ls -1 /usr/local/)" && echo $file

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-numpy \
    zip \    
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install python3.6 libpython3.6

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ARG PYTHON=python3
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ${PIP} install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN ${PIP} install torchvision

RUN pip install cupy-cuda100
RUN pip install pycuda

# Install NVIDIA Rapids and it's deps
RUN apt install libopenblas-base libomp-dev -y
RUN pip install cugraph-cuda100
RUN pip install cuml-cuda100
RUN pip install cudf-cuda100==0.6

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN pip install tensorflow-gpu

#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc

RUN ${PIP} --no-cache-dir install jupyter matplotlib pyinstrument
# RUN ${PIP} install jupyter matplotlib o pencv-python opencv-contrib-python pyinstrument

# Core linux dependencies. 
RUN apt-get install -y --fix-missing \
        build-essential \
        cmake \
    	curl \
	gfortran \
    	graphicsmagick \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
	libgraphicsmagick1-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
	libgraphicsmagick1-dev \
	libavcodec-dev \
	libboost-all-dev \
	libgtk2.0-dev \
	liblapack-dev \
        liblapacke-dev \
	libswscale-dev \
	libcanberra-gtk-module \
        libboost-dev \
        libboost-all-dev \
        libeigen3-dev \
        # python3.6 \
	python3-dev \
	python3-numpy \
	python3-scipy \
	software-properties-common \
	zip \
        vim \
        qt5-default \
	&& apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN ${PIP} --no-cache-dir install \
    hdf5storage \
    h5py \
    py3nvml \
    scikit-image \
    scikit-learn

WORKDIR /
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.0.1.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.0.1.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN mv opencv-4.0.1 opencv
RUN mv opencv_contrib-4.0.1 opencv_contrib
RUN mkdir /opencv/build
WORKDIR /opencv/build


RUN cmake -DBUILD_TIFF=ON \
		  -DBUILD_opencv_java=OFF \
		  -DWITH_CUDA=ON \
		  -DENABLE_FAST_MATH=1 \
		  -DCUDA_FAST_MATH=1 \
		  -DWITH_CUBLAS=1 \
		  -DENABLE_AVX=ON \
		  -DWITH_OPENGL=ON \
		#   -DWITH_OPENCL=OFF \
		  -DWITH_IPP=ON \
		  -DWITH_TBB=ON \
		  -DWITH_EIGEN=ON \
		  -DWITH_V4L=ON \
		#   -DBUILD_TESTS=OFF \
		#   -DBUILD_PERF_TESTS=OFF \
		  -DCMAKE_BUILD_TYPE=RELEASE \
		  -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
		  -DPYTHON_EXECUTABLE=$(which python) \
		  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
		  -DPYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
                  -DINSTALL_PYTHON_EXAMPLES=ON \
                  -DINSTALL_C_EXAMPLES=OFF \
                  -DOPENCV_ENABLE_NONFREE=ON \
                  -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
                  -DBUILD_EXAMPLES=ON \
                  -D CUDA_TOOLKIT_ROOT_DIR=/opt/cuda/9.0 \
                  -DWITH_QT=ON ..
RUN make -j4 \
        && make install \
	&& rm /opencv.zip \
        && rm /opencv_contrib.zip \
	&& rm -rf /opencv \
        && rm -rf /opencv_contrib

WORKDIR /

# dlib
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.16' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA --clean


RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_restore_models.ipynb

RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /tf
EXPOSE 8888 6006

RUN useradd -ms /bin/bash container_user

RUN ${PYTHON} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://localhost:8888'"]
