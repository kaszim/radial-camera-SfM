# radial-camera-SfM
A proof of concept SfM pipeline using the radial camera model

# How to build
You will need the following packages on your system:
* ceres-solver
* eigen3
* libomp

Then just run `cmake .` and then `make -j4` and the executable will be created.

# How to run
To run the executable on the provided example data, simply run:
```bash 
./sfm ExampleData/data_pumpkin 1 100 196 8 > results
```
