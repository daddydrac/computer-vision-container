### Machine Vision Container: Docker, TensorFlow, TensorRT, PyTorch, Rapids.ai: CuGraph, CuML, and CuDF, CUDA 10, OpenCV, CuPy, and PyCUDA ###

GPU Acclerated computing container for computer vision applications, that are reproducible across environments.

-----------------------------------------------------------

### Features ###

- Contains most of the features in NVIDIA [Rapids.ai](https://rapids.ai/)

- cuDF is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data.

- cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects.

- cuGraph library is a collection of graph analytics that allow you to vizualize what is going on inside graph-based neural networks.

- Learn to vizualize what is going on inside graph-based neural networks with cuGraph, [see here for more](https://medium.com/rapids-ai/rapids-cugraph-1ab2d9a39ec6?ncid=em-ele-n2-79899&ncid=so-lin-lt-798&_lrsc=af6f9f62-bebc-4a4d-8008-1a4141ce62e2)!

- NVIDIA TensorRT inference accelerator and CUDA 10

- PyTorch 1.0

- PyCUDA 2018.1.1

- CuPy:latest 

- Tensorflow for GPU v1.13.1 & TensorBoard

- OpenCV v4.0.1 for GPU

- Ubuntu 18.04 so you can 'nix your way through the cmd line!

- cuDNN7.4.1.5 for deeep learning in CNN's

- Hot Reloading: code updates will automatically update in container from /apps folder.

- TensorBoard is on localhost:6006 and GPU enabled Jupyter is on localhost:8888.

- Python 3.6.7

- Only Pascal and Turing arch are supported 

-------------------------------------------------------------


### Before you begin (This might be optional) ###

Link to nvidia-docker2 install: [Tutorial](https://medium.com/@sh.tsang/docker-tutorial-5-nvidia-docker-2-0-installation-in-ubuntu-18-04-cb80f17cac65)

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


### Step 3: Check to make sure GPU drivers and CUDA is running ###

- Exec into the container and check if your GPU is registering in the container and CUDA is working:

- Get the container id:

` docker ps `

- Exec into container:

` docker exec -u root -t -i <container id> /bin/bash `

- Check if NVIDIA GPU DRIVERS have container access:

` nvidia-smi `

- Check if CUDA is working:

` nvcc -V `


### Step 4: How to launch TensorBoard ###

(It helps to use multiple tabs in cmd line, as you have to leave at least 1 tab open for TensorBoard@:6006)

- Demonstrates the functionality of TensorBoard dashboard


- Exec into container if you haven't, as shown above:


- Get the `<container id>`:
 

` docker ps `


` docker exec -u root -t -i <container id> /bin/bash `


- Then run in cmd line:


` tensorboard --logdir=//tmp/tensorflow/mnist/logs `


- Type in: ` cd / ` to get root.

Then cd into the folder that hot reloads code from your local folder/fav IDE at: `/apps/apps/gpu_benchmarks` and run:


` python tensorboard.py `


- Go to the browser and navigate to: ` localhost:6006 `



### Step 5: Run tests to prove container based GPU perf ###

- Demonstrate GPU vs CPU performance:

- Exec into the container if you haven't, and cd over to /tf/notebooks/apps/gpu_benchmarks and run:

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





