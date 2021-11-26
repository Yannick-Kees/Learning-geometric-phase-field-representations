from pointclouds import *

def draw_phase_field(f,x_,y_):
    xlist = np.linspace(0, x_, 100)
    ylist = np.linspace(0, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)

    Z = np.zeros((len(X),len(X[0])))
    """
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j]= f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]
    """
    Z = [[ f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]  for i in range(len(X))] for j in range(len(X[0]))]
    plt.figure()

    levels = [-1000.0,-5.0,-.5,0.0,.5,200.0]
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    plt.show()
    
def color_plot(f):
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
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return

