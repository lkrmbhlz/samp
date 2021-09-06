# samp - Simplified Asymmetry Measure for Point Cloud Projections

Samp is an implementation of a simple grid based asymmetry measure on 2D projections of point cloud data objects. 

## Method

- Project 3D point cloud data onto a 2-dimensional subspace which centers the object and aligns it along the two axes. (While any type of projection (e.g. PCA) can be used, we found that it performs best on varimax projections.)
- Walk along each of the resulting axes in a grid-like manner and collect a value which approximates the axis-asymmetry values in each bin of the object.
The asymmetry-value at each step is the difference between the distance of the maximum and the minimum value of points in the bin along the axis. 
- The resulting value, which approximates how asymmetric an object is, is an 
arbitrary positive value. It ranges from zero to the span of the object's projection times the step size.
- If you want to cluster your objects according to their symmetry, we propose using the minimum and the maximum 
of the asymmetries of the axes as representation for each object.

## Visualization
A plotting mechanism is provided in the symmetry functions to visualize the result, in this case on a varimax projection 
of a 3D-scan of a talus bone:

![plot](images\asymmetry_example_01.PNG)

The green dots denote the maximum value in each bin and the red dots the corresponding minimum.
The graph on the right visualizes the asymmetry value.


## Examples
Jupyter notebook examples can be found in [the samp folder.](samp)

