## Machine Vision Container ##

Reproduce computer vision applications across environments while having containerized access to NVIDIA GPU's
 
-----------------------------------------------------------

- NVIDIA TensorRT inference accelerator and CUDA 10

- PyTorch 1.0

- PyCUDA 2018.1.1

- CuPy:latest 

- Tensorflow for GPU v1.13.1 

- TensorBoard by TensorFlow

- Python 3.6

- OpenCV v4.0.1 for GPU

- Ubuntu 18.04 so you can 'nix your way through the cmd line!

- CUDA 10.0

- cuDNN7.4.1.5 for deeep learning in CNN's

- Hot Reloading: code updates will automatically update in container.

- TensorBoard is on localhost:6006 and GPU enabled Jupyter is on localhost:8888.

-------------------------------------------------------------


### Before you begin (This might be optional) ###

You must install nvidia-docker2 and all it's deps first, assuming that is done, run:


 ` sudo apt-get install nvidia-docker2 `
 
 ` sudo pkill -SIGHUP dockerd `
 
 ` sudo systemctl daemon-reload `
 
 ` sudo systemctl restart docker `
 

How to run this container:


### Step 1 ###

` docker build -t <container name> . `  < note the . after <container name>


### Step 2 ###

Run the image, mount the volumes for Jupyter and app folder for your fav IDE, and finally the expose ports `8888` for TF1, and `6006` for TensorBoard.


` docker run --rm -it --runtime=nvidia --user $(id -u):$(id -g) --group-add container_user --group-add sudo -v "${PWD}:/apps" -v $(pwd):/tf/notebooks  -p 8888:8888 -p 0.0.0.0:6006:6006  <container name> `


### Step 2: Check to make sure GPU drivers and CUDA is running ###

- Exec into the container and check if your GPU is registering in the container and CUDA is working:

- Get the container id:

` docker ps `

- Exec into container:

` docker exec -u root -t -i <container id> /bin/bash `

- Check if NVIDIA GPU DRIVERS have container access:

` nvidia-smi `

- Check if CUDA is working:

` nvcc -V `


### Step 3: How to launch TensorBoard ###

- Demonstrates the functionality of TensorBoard dashboard


- Exec into container if you haven't:


` docker ps `  ( gets <container id> ).


` docker exec -u root -t -i <container id> /bin/bash `


- Then run in cmd line:


` tensorboard --logdir=/tmp/tensorflow/logs/ `


- cd over to /tf/notebooks/tf/apps/gpu_benchmarks and run:


` python tensorboard.py `


- Go to the browser and navigate to: ` localhost:6006 `



### Step 4: Run tests to prove container based GPU perf ###

- Demonstrate GPU vs CPU performance:

- cd over to /tf/notebooks/tf/apps/gpu_benchmarks and run:

- CPU Perf:

` python benchmark.py cpu 10000 `

- CPU perf should return something like this:

`Shape: (10000, 10000) Device: /cpu:0
Time taken: 0:00:03.934996`

- GPU perf:

` python benchmark.py gpu 10000 `

- GPU perf should return something like this:

`Shape: (10000, 10000) Device: /gpu:0
Time taken: 0:00:01.032577`


--------------------------------------------------


### Known conflicts with nvidia-docker and Ubuntu ###

AppArmor on Ubuntu has sec issues, so remove docker from it on your local box, (it does not hurt security on your computer):

` sudo aa-remove-unknown `


## END ##

