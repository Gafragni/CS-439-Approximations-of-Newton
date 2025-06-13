# CS-439-Approximations-of-Newton

# Approximations of Newton's method
In this code we explore the different ways to approximate Newton's method in order to reduce the requirements on the function. Newton's method requires the function to be twice differentiable which is not always the case. So here we will explore some approximations of this method that do not require the hessian, and this code base is split in three main parts depending on the requirements.

The code is split according to the requirements on the function :
- 1 dimensional function, differentiable => First order methods, in file FirstOrderNewton1D.ipynb
- 1 dimensional function, not differentiation => Zero order methods, in file ZeroOrderNewton1D.ipynb
- 2 dimensional function, not differentiable => Coordinate descent with Newton-like steps, in MultiDimApprox.ipynb

There is also a helper file, function_examples.py, which defines some funtions to test the algorithms and plot the results.

### FirstOrderNewton1D.ipynb
The main idea behind the methods tried in this file are based on the secant method, which approximates the hessian using finite differences of gradients (1D).

### ZeroOrderNewton1D.ipynb
The main idea behind the zero order approximations of Newton's method is to approximate the gradient and hessian using finite differences (1D).

### MultiDimApprox.ipynb
In order to keep the number of evaluations independant from the dimension of the problem, we can base the minimization on coordinate descent which reduces the problem to 1D, and we can use the previous ideas from 1D to do Newton-like steps for coordinate descent.
