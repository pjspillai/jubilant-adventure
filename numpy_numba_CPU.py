#!/usr/bin/env python
# coding: utf-8

# # GTC 2019 Numba Tutorial Notebook 1: NumPy and Numba on the CPU

# ## Table of Contents

# - 1 - NumPy and Numba on the CPU
# - [2 - CuPy and Numba on the GPU](2%20-%20CuPy%20and%20Numba%20on%20the%20GPU.ipynb)
# - [3 - Memory Management](3%20-%20Memory%20Management.ipynb)
# - [4 - Writing CUDA Kernels](4%20-%20Writing%20CUDA%20Kernels.ipynb)
# - [5 - Troubleshooting and Debugging](5%20-%20Troubleshooting%20and%20Debugging.ipynb)
# - [6 - Extra Topics](6%20-%20Extra%20Topics.ipynb)

# Although this is a GPU tutorial, it helps to start with some fundamentals on the CPU. (Don't worry, they will carry over to the GPU.  We promise!)  Following the tips in this notebook in your CPU code will put you in a good position to GPU-accelerate the parts that need it.
# 
# 
# ## What is NumPy?
# 
# NumPy is a very popular Python library for efficiently working with large amounts of numerical data.  Even if you haven't used NumPy directly, many popular libraries depend on NumPy, such as pandas, scikit-learn, matplotlib, and statsmodels.  Learning to use NumPy is *the most important thing you can do to speed up your numerical applications*.
# 
# Although this tutorial is not an introduction to NumPy, we want to stop and reflect on the three major components of NumPy.  We will see them in the next notebook on the GPU.

# In[ ]:


import numpy as np


# ### 1. A multidimensional, homogeneous array type, along with a type system for array elements (called "dtypes")
# 
# A NumPy array is described by several attributes (skipping a few for clarity):
# 
# * data: a pointer to a data buffer with the actual array values
# * dtype: the data type of each array element.  Ex: `float32`, `int64`, `complex128`
# * shape: size of each array dimension.  Ex: 1D = `(4,)`, 2D = `(6,8)`, 3D = `(2, 4, 8)`
# * strides: the number of bytes that separate elements as you move along each dimension
# 
# NumPy offers a number of functions for allocating arrays

# In[ ]:


x = np.zeros(shape=(2,3), dtype=np.float64)
print(repr(x))
print(x.dtype)
print(x.shape)
print(x.strides)


# In[ ]:


y = np.ones(shape=(2,3,4), dtype=np.int32)
y


# In[ ]:


# empty doesn't initialize the contents of the array, so only use this
# if you are planning to overwrite every element yourself!
z = np.empty(shape=10, dtype=np.complex128)
z


# Note that slicing of NumPy arrays (like a Python list) results in a *view* on the array.  This makes slicing very fast and memory efficient, but be careful if you change the contents of a view, the original array changes as well:

# In[ ]:


orig = np.arange(20)
print(orig)
print(orig.strides)


# In[ ]:


view = orig[::3]  # every third element
print(view)
print(view.strides)
view[1] = 99
print('view:', view)
print('original array:', orig)


# ### 2. A "universal function" paradigm for operating on arrays with different numbers of dimensions
# 
# NumPy defines a "universal function" ("ufunc" for short) to be a function that operates on each element in an array, or combine single elements from several input arrays.  A ufunc takes as inputs arrays with different numbers of dimensions, or even scalar values, and returns a new array.  The process by which array elements are matched up is called *broadcasting*.
# 
# It is probably easiest to show what happens by example.  We'll use the NumPy `add` ufunc to demonstrate what happens:

# In[ ]:


import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

np.add(a, b)


# Ufuncs also can combine scalars with arrays:

# In[ ]:


np.add(a, 100)


# Arrays of different, but compatible dimensions can also be combined.  The lower dimensional array will be replicated along additional axes to match the dimensionality of the higher dimensional array:

# In[ ]:


c = np.arange(4*4).reshape((4,4))
print('c:', c)

np.add(b, c)


# In the above situation, the `b` array is added to each row of `c`.  If we want to add `b` to each column, we need to transpose it.  There are several ways to do this, but one way is to insert a new axis using `np.newaxis`:

# In[ ]:


b_col = b[:, np.newaxis]
b_col


# In[ ]:


np.add(b_col, c)


# The NumPy documentation has a much more extensive discussion of ufuncs:
# 
# https://docs.scipy.org/doc/numpy/reference/ufuncs.html

# ### 3. A large library of array functions
# 
# In addition to all the standard basic math operations (+,-,\*,/), NumPy offers many additional classes of functions:
# 
# * Lineary algebra
# * Special math functions (trig, exp/log, polynomials)
# * Cumulative functions
# * Logical (bool) operations
# * Random number generation
# 
# Most of these functions are implemented using compiled C code, so they execute much faster than regular Python code. It is a good idea to be familiar with the array functions that NumPy offers so you don't reinvent the wheel in your own code.

# ### What are NumPy's limitations?
# 
# NumPy is great, but there are some limitations.  Not every equation has been implemented as a fast NumPy ufunc already (especially if you are creating something new!).  When you need to go beyond what NumPy offers, you will have to fall back to Python, which will expose one of the following issues:
# 
# * Looping over individual array elements in Python is very slow.
# 
# When you need to write your own operations on NumPy arrays, doing it with for loops can be 100x slower than a native NumPy function.  For example, if you were implementing Conway's Game of Life, this very straightforward code (assume `count_neighbors()` is implemented elsewhere) would have poor performance:
# 
# ``` python
# def life_step(state):
#     new_state = np.empty_like(state)
#     
#     for i in range(new_state.shape[0]):
#         for j in range(new_state.shape[1]):
#             nbrs_count = count_neighbors(state, i, j)
#             if nbrs_count == 3 or (nbrs_count == 2 and state[i,j]):
#                 new_state[i,j] = True
#             else:
#                 new_state[i,j] = False
#                 
#     return new_state
# ```
# 
# * Combining several NumPy ufuncs into a large expression can be both hard to read, and still too slow.
# 
# Clever users of NumPy will find ways to subvert the looping issues by creating "NumPy haikus", which are terse combinations of several NumPy functions that accomplish the end goal without writing a loop.  My favorite example of this is the [Jake VanderPlas implementation of the Game of Life](https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/) time step function:
# 
# ``` python
# def life_step_1(X):
#     """Game of life step using generator expressions"""
#     nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
#                      for i in (-1, 0, 1) for j in (-1, 0, 1)
#                      if (i != 0 or j != 0))
#     return (nbrs_count == 3) | (X & (nbrs_count == 2))
# ```
# 
# A beginning NumPy user will have no idea what is going on here, but this does in fact achieve the goal.  It also uses NumPy functions to do the majority of the looping, so it is easily 10x faster than the previous implementation.  However, this code results in the creation of at least 14 temporary arrays the size of the game board.  For performance critical code, that can still be a problem.
# 
# * NumPy does not use the parallel execution capabilities of your computer.
# 
# NumPy functions are (with some exceptions) not going to use multiple CPU cores, never mind the GPU.  You are on your own for parallelizing much of your code.
# 
# 
# All of the above limitations can be addressed by combining NumPy with another package, called *Numba*, which we'll talk about next.

# ## What is Numba?
# 
# Numba is a **just-in-time**, **type-specializing**, **function compiler** for accelerating **numerically-focused** Python.  That's a long list, so let's break down those terms:
# 
#  * **function compiler**: Numba compiles Python functions, not entire applications, and not parts of functions.  Numba does not replace your Python interpreter, but is just another Python module that can turn a function into a (usually) faster function. 
#  * **type-specializing**: Numba speeds up your function by generating a specialized implementation for the specific data types you are using.  Python functions are designed to operate on generic data types, which makes them very flexible, but also very slow.  In practice, you only will call a function with a small number of argument types, so Numba will generate a fast implementation for each set of types.
#  * **just-in-time**: Numba translates functions when they are first called.  This ensures the compiler knows what argument types you will be using.  This also allows Numba to be used interactively in a Jupyter notebook just as easily as a traditional application
#  * **numerically-focused**: Currently, Numba is focused on numerical data types, like `int`, `float`, and `complex`.  There is very limited string processing support, and many string use cases are not going to work well on the GPU.  To get best results with Numba, you will likely be using NumPy arrays.
# 
# ### Requirements
# 
# Numba supports a wide range of operating systems:
# 
#  * Windows 7 and later, 32 and 64-bit
#  * macOS 10.9 and later, 64-bit
#  * Linux (most anything >= RHEL 5), 32-bit and 64-bit
# 
# and Python versions:
# 
#  * Python 2.7, 3.4-3.7
#  * NumPy 1.10 and later
# 
# and a very wide range of hardware:
# 
# * x86, x86_64/AMD64 CPUs
# * NVIDIA CUDA GPUs (Compute capability 3.0 and later, CUDA 8.0 and later)
# * AMD GPUs (ROCm on Linux)
# * ARM 32-bit (Raspbery Pi) and 64-bit (Jetson TX2)
# * POWER8/9
# 
# For this tutorial, we will be using Linux 64-bit and CUDA 8.

# ### First Steps
# 
# Let's write our first Numba function and compile it for the **CPU**.  The Numba compiler is typically enabled by applying a *decorator* to a Python function.  Decorators are functions that transform Python functions.  Here we will use the CPU compilation decorator:

# In[ ]:


from numba import jit
import math

@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)


# The above code is equivalent to writing:
# ``` python
# def hypot(x, y):
#     x = abs(x);
#     y = abs(y);
#     t = min(x, y);
#     x = max(x, y);
#     t = t / x;
#     return x * math.sqrt(1+t*t)
#     
# hypot = jit(hypot)
# ```
# This means that the Numba compiler is just a function you can call whenever you want!
# 
# Let's try out our hypotenuse calculation:

# In[ ]:


hypot(3.0, 4.0)


# The first time we call `hypot`, the compiler is triggered and compiles a machine code implementation for float inputs.  Numba also saves the original Python implementation of the function in the `.py_func` attribute, so we can call the original Python code to make sure we get the same answer:

# In[ ]:


hypot.py_func(3.0, 4.0)


# ### Benchmarking
# 
# An important part of using Numba is measuring the performance of your new code.  Let's see if we actually sped anything up.  The easiest way to do this in the Jupyter notebook is to use the `%timeit` magic function.  Let's first measure the speed of the original Python:

# In[ ]:


get_ipython().magic('timeit hypot.py_func(3.0, 4.0)')


# The `%timeit` magic runs the statement many times to get an accurate estimate of the run time.  It also returns the best time by default, which is useful to reduce the probability that random background events affect your measurement.  The best of 3 approach also ensures that the compilation time on the first call doesn't skew the results:

# In[ ]:


get_ipython().magic('timeit hypot(3.0, 4.0)')


# Numba did a pretty good job with this function.  It's 3x faster than the pure Python version.
# 
# Of course, the `hypot` function is already present in the Python module:

# In[ ]:


get_ipython().magic('timeit math.hypot(3.0, 4.0)')


# Python's built-in is even faster than Numba!  This is because Numba does introduce some overhead to each function call that is larger than the function call overhead of Python itself.  Extremely fast functions (like the above one) will be hurt by this.
# 
# (However, if you call one Numba function from another one, there is very little function overhead, sometimes even zero if the compiler inlines the function into the other one.)

# ### How does Numba work?
# 
# The first time we called our Numba-wrapped `hypot` function, the following process was initiated:
# 
# ![Numba Flowchart](img/numba_flowchart.png "The compilation process")
# 
# We can see the result of type inference by using the `.inspect_types()` method, which prints an annotated version of the source code:

# In[ ]:


hypot.inspect_types()


# Note that Numba's type names tend to mirror the NumPy type names, so a Python `float` is a `float64` (also called "double precision" in other languages).  Taking a look at the data types can sometimes be important in GPU code because the performance of `float32` and `float64` computations will be very different on CUDA devices.  An accidental upcast can dramatically slow down a function.

# ### When Things Go Wrong
# 
# Numba cannot compile all Python code.  Some functions don't have a Numba-translation, and some kinds of Python types can't be efficiently compiled at all (yet).  For example, Numba does not support dictionaries (as of this tutorial):

# In[ ]:


@jit
def cannot_compile(x):
    return x['key']

cannot_compile(dict(key='value'))


# Wait, what happened??  By default, Numba will fall back to a mode, called "object mode," which does not do type-specialization.  Object mode exists to enable other Numba functionality, but in many cases, you want Numba to tell you if type inference fails.  You can force "nopython mode" (the other compilation mode) by passing arguments to the decorator:

# In[ ]:


@jit(nopython=True)
def cannot_compile(x):
    return x['key']

cannot_compile(dict(key='value'))


# Now we get an exception when Numba tries to compile the function, with an error that says:
# ```
# - argument 0: cannot determine Numba type of <class 'dict'>
# ```
# which is the underlying problem.
# 
# We will see other `@jit` decorator arguments in future sections.

# # Exercise
# 
# Below is a function that loops over two input NumPy arrays and puts their sum into the output array.  Modify this function to call the `hypot` function we defined above.  We will learn a more efficient way to write such functions in a future section.
# 
# (Make sure to execute all the cells in this notebook so that `hypot` is defined.)

# In[ ]:


@jit(nopython=True)
def ex1(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


# In[ ]:


in1 = np.arange(10, dtype=np.float64)
in2 = 2 * in1 + 1
out = np.empty_like(in1)

print('in1:', in1)
print('in2:', in2)

ex1(in1, in2, out)

print('out:', out)


# In[ ]:


# This test will fail until you fix the ex1 function
np.testing.assert_almost_equal(out, np.hypot(in1, in2))


# In[ ]:





# ## Next

# Please continue to the next notebook [2 - CuPy and Numba on the GPU](2%20-%20CuPy%20and%20Numba%20on%20the%20GPU.ipynb).
