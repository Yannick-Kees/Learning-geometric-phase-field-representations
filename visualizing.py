from pointclouds import *

def draw_phase_field(f,x_,y_):
    # Creating Contour plot of f
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]

    xlist = np.linspace(0, x_, 100)
    ylist = np.linspace(0, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)

    Z = [[ f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]  for i in range(len(X))] for j in range(len(X[0]))] # Evaluate function in points
    
    plt.figure()                                                    # Draw contour plot
    levels = [-1000.0,-5.0,-.5,0.0,.5,200.0]                        # Specify contours to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    plt.show()
    return
    
def color_plot(f):
    # Creating 3D plot of f on [0,1]^2
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(0, 1, 0.01)
    Y = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(X),len(X[0])))

    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j]= f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}') # <- This may or may not be out commented, depending on compiler

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return

def draw_point_cloud(pc):
    # Plotting point cloud
    
    # Parameters:
    #   pc:      Tensor of points
    
    pointcloud = pc.detach().numpy().T 
    plt.plot(pointcloud[0],pointcloud[1], '.')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    return


# draw_point_cloud(triangle)
    
