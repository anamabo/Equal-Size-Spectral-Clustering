# Equal-Size Spectral Clustering
This is a modification of the spectral clustering code that builds clusters balanced 
in the number of points. A detailed explanation of this algorithm can be found 
[in this Medium blog post](https://medium.com/p/cce65c6f9ba3/edit).

## Prerequisities
You need to install Python 3.9. There is a Pipfile to install the required libraries.

## Toy datasets
In the folder `datasets` we have provided you with a toy dataset
so you can run the clustering code right away. Specifications of the input dataset
are explained in the blog post. 

## Examples
* example1.py: From a set of hyperparameters, you obtain clusters with sizes roughly equal to N / `nclusters`  
* example2.py: From a range of cluster sizes, you obtain the clusters hyperparameters to run the clustering code. 
