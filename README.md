# radial-camera-SfM
A proof of concept SfM pipeline using the radial camera model

# How to build
You will need the following packages on your system:
* ceres-solver (tested with 1.14.0)
* eigen3 (tested with 3.3.7)
* libomp (tested with 8.0.0)

Then just run `cmake .` and then `make -j4` and the executable will be created.

# How to run
To run the executable on the provided example data, simply run:
```bash 
./sfm ExampleData/data_church 1 100 92 8 > results
./sfm $data_folder $threshhold_RANSAC $iterations_RANSAC $num_images $min_num_features
```
