# K-Lines clustering - Line Detection
> By Frank Lai, Nick Shaju

## How to run
Install scikit spatial
```
pip install scikit-spatial
```
Then run ```tests.py``` as an executable with the command line arguments below

## Command Line Arguments
```
test: Number id of the test. Currently 3 tests supported with ids [1,2,3]. Required

-i, --iterations: Number of iterations to run the fit function. Default: 20

-o, --output: if used, test point cloud data will be outputted to a txt file in the format x,y,z entries per row. Default: False

-v: if used, outputs figures and text for each single iteration. Default: False
```
Example, to run test 2 for 30 iterations

```
./tests