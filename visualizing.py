from pointclouds import *


#############################
# 2D point cloud ############
#############################


def draw_phase_field(f,x_,y_, i, film):
    # Creating Contour plot of f
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)
    alpha = np.pi *1./3.
    Z = [[ f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]  for j in range(len(X[0]))  ] for i in range(len(X)) ] # Evaluate function in points
    
    fig = plt.figure()                                                    # Draw contour plot
    levels = [-1000.0,-5.0,-.5,0.0,.5,200.0]                        # Specify contours/level set to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    if film:
        plt.savefig("images/mov/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return


def draw_height(f):
    x = np.linspace(0,2*np.pi,500)
    y = [ f(Tensor([ .3 * np.sin(a), .3 * np.cos(a) ]  )).detach().numpy()[0] for a in x     ]
    plt.xlabel("Angle")
    plt.ylabel("Function value")
    plt.plot(x,y)
    plt.show()
    
def color_plot(f, y, film):
    # Creating 3D plot of f on [0,1]^2
        
    # Parameters:
    #   f:      Function to plot
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-.5, .5, 0.01)
    Y = np.arange(-.5, .5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(X),len(X[0])))
    alpha = np.pi *1./3.
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i][j]= f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]
    # f(Tensor([ X[i][j], Y[i][j] ] ))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}') # <- This may or may not be out commented, depending on compiler

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if film:
        plt.savefig('images/mov/cp' + str(y).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return

#############################
# 2/3D point cloud ##########
#############################



def draw_point_cloud(pc):
    # Plotting point cloud
    
    # Parameters:
    #   pc:      Tensor of points

    d = pc.shape[1] # dimension
    
    if (d==2):
        pointcloud = pc.detach().numpy().T 
        plt.plot(pointcloud[0],pointcloud[1], '.')
        plt.xlim(-.5,.5)
        plt.ylim(-.5,.5)
        plt.show()
        return
    if (d==3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pointcloud = pc.detach().numpy().T 
        ax.scatter(pointcloud[0],pointcloud[1],pointcloud[2])
        plt.show()

        
        
#############################
# 3D point cloud ############
#############################       

def plot_implicit(fn, shift=True):
    # Creating 3D contour plot of f on [0,1]^2 using marching cubes 
        
    # Parameters:
    #   fn:      Function to plot
    if shift:
        xa = ya = za= -.3
        xb = yb = zb = .3
    else:
        xa = ya = za= 0.0
        xb = yb = zb = 1.0
    """
    xa = -0.4
    xb = .1
    ya = -.1
    yb  = 0.4
    za = -.4
    zb = .1
    """
    plot = k3d.plot()
    x = np.linspace(xa, xb, 60, dtype=np.float32)
    y = np.linspace(ya, yb, 60, dtype=np.float32)
    z = np.linspace(za,zb, 60, dtype=np.float32)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    Z = [[[ fn(Tensor([ x[i][j][k], y[i][j][k], z[i][j][k] ] )).detach().numpy()  for k in range(len(x[0][0]))  ] for j in range(len(x[0])) ] for i in range(len(x)) ]# Evaluate function in points
    plt_iso = k3d.marching_cubes(Z, compression_level=5, xmin=xa, xmax=xb,ymin=ya, ymax=yb,  zmin=za, zmax=zb, level=0.0, flat_shading=False)
    plot += plt_iso
    #plot += plt_iso
    plot.display()



def test_f(t):
    # test function for implicit plotting
    return t[0]**2+t[1]**2+t[2]**2 -1 


def toParaview(f, n):
    # Dimensions 
    # 
    nx, ny, nz = n, n, n
    lx, ly, lz = .6, .6, .6
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    ncells = nx * ny * nz 
    npoints = (nx + 1) * (ny + 1) * (nz + 1) 

    # Coordinates 
    # 
    X = np.arange(0, lx + 0.1*dx, dx, dtype='float64') -.3
    Y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')  -.3
    Z = np.arange(0, lz + 0.1*dz, dz, dtype='float64') -.3
    x = np.zeros((nx + 1, ny + 1, nz + 1)) 
    y = np.zeros((nx + 1, ny + 1, nz + 1)) 
    z = np.zeros((nx + 1, ny + 1, nz + 1)) 

    values = np.zeros((nx + 1, ny + 1, nz + 1)) 
    values = np.array([X,Y,Z]).T
   
      
    # 
    for k in range(nz + 1): 
        report_progress(k, nz  , 0 )
        for j in range(ny + 1):
            for i in range(nx + 1): 
                x[i,j,k] = X[i] 
                y[i,j,k] = Y[j] 
                z[i,j,k] = Z[k]
                
                
              
    # Variables 
    
    Z = np.array([ f(Tensor([ x[i][j][k], y[i][j][k], z[i][j][k] ] ).to(device)).detach()  for k in range(len(x[0][0]))  for j in range(len(x[0])) for i in range(len(x)) ])

    #pressure = np.random.rand(ncells).reshape( (nx, ny, nz)) 
    #temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1)) 
    structuredToVTK("./structured"+str(n), x, y, z,  pointData = {"NN" : Z})

