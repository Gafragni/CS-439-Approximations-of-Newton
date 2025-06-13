import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import some colors for the plots
color_black = mcolors.CSS4_COLORS['black']
color_blue = mcolors.TABLEAU_COLORS['tab:blue']
color_orange = mcolors.TABLEAU_COLORS['tab:orange']
color_green = mcolors.TABLEAU_COLORS['tab:green']
color_red = mcolors.TABLEAU_COLORS['tab:red']

# A convex function in 1D, with its gradient and hessian
def f1d(x):
    return np.exp(0.5 * x) - x
def f1d_grad(x):
    return 0.5 * np.exp(0.5 * x) - 1
def f1d_hess(x):
    return 0.25 * np.exp(0.5 * x)

# A non-convex but smooth function in 1D (with global minmum), with its gradient and hessian
def f1d_nonconvex(x):
    return (0.01) * x*x*x*x + (-0.02) * x*x*x + (-0.2) * x*x + (0.5) * x + (1.0)
def f1d_nonconvex_grad(x):
    return 4*(0.01) * x*x*x + 3*(-0.02) * x*x + 2*(-0.2) * x + (0.5)
def f1d_nonconvex_hess(x):
    return 12*(0.01) * x*x + 6*(-0.02) * x + 2*(-0.2)

# A convex function in higher dimension, with its gradient and hessian
def fnd(x):
    return np.sum(np.exp(0.5 * x) - x, axis=-1)
def fnd_grad(x):
    return 0.5 * np.exp(0.5 * x) - 1
def fnd_hess(x):
    return np.diagflat(0.25 * np.exp(0.5 * x))

# Plot the 1D function with its gradient and hessian (on the given points)
def plot_function_1d(f, f_grad, f_hess, x):
    plt.plot(x, f(x), linestyle='-', color=color_blue, label='f1d')
    plt.plot(x, f_grad(x), linestyle='--', color=color_orange, label='f1d_grad')
    plt.plot(x, f_hess(x), linestyle=':', color=color_green, label='f1d_hess')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid(linestyle=':')
    plt.legend()
    plt.show()

# Plot the 1D function with its gradient and hessian (on the given points)
# And add the path of the convergence (output of the chosen method)
def plot_convergence_1d(f, f_grad, f_hess, x, x_result, y_result, label):
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot(x, f(x), linestyle='-', color=color_blue, label='f')
    plt.plot(x, f_grad(x), linestyle='--', color=color_orange, label='f_grad')
    plt.plot(x, f_hess(x), linestyle=':', color=color_green, label='f_hess')
    plt.plot(x_result, y_result, linestyle='-.', color=color_black, label=label)
    plt.scatter(x_result, y_result, color=color_black)
    plt.grid(linestyle=':')
    plt.legend()
    #plt.savefig('newton1D.pdf', bbox_inches="tight")
    plt.show()

# Plot the 2D function
def plot_function_2d(f=fnd, mesh_width=5):
    X = np.arange(-mesh_width, mesh_width, 0.1)
    Y = np.arange(-mesh_width, mesh_width, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.stack((X, Y), axis=-1))
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(azim=-115, elev=30)
    ax.set_box_aspect(None, zoom=0.85)
    ax.plot_surface(X, Y, Z, edgecolor=color_blue, lw=0.5, rstride=8, cstride=8, alpha=0.2)
    ax.contourf(X, Y, Z, levels=10, offset=-5, cmap='coolwarm')
    ax.set(xlim=(-mesh_width, mesh_width), ylim=(-mesh_width, mesh_width), zlim=(-5, 15), xlabel='X', ylabel='Y', zlabel='Z')
    plt.show()

# Plot the 2D function and add the path of the convergence (output of the chosen method)
def plot_convergence_2d(pos_result, val_result, f=fnd, mesh_width=5):
    X = np.arange(-mesh_width, mesh_width, 0.1)
    Y = np.arange(-mesh_width, mesh_width, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.stack((X, Y), axis=-1))
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(azim=-115, elev=30)
    ax.set_box_aspect(None, zoom=0.85)
    ax.plot_surface(X, Y, Z, edgecolor=color_blue, lw=0.5, rstride=8, cstride=8, alpha=0.2)
    ax.plot(pos_result[::,0], pos_result[::,1], val_result, linestyle='-.', color=color_black, alpha=1.0)
    ax.scatter(pos_result[::,0], pos_result[::,1], val_result, color=color_black, alpha=1.0)
    ax.contourf(X, Y, Z, levels=10, offset=-5, cmap='coolwarm')
    ax.set(xlim=(-mesh_width, mesh_width), ylim=(-mesh_width, mesh_width), zlim=(-5, 15), xlabel='X', ylabel='Y', zlabel='Z')
    #plt.savefig('newton2D.pdf', bbox_inches="tight")
    plt.show()