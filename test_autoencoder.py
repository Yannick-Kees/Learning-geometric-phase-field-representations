from shapemaker import *

def test_shape(index):
    # index:    Index in dataset which to test
    
    autoencoder = PCAutoEncoder64(3, 1000)

    autoencoder.load_state_dict(torch.load(r"autoencoder64.pth", map_location=device))
    autoencoder.eval()
    dataset = np.load(open("dataset1k.npy", "rb"))

    points= [dataset[index],dataset[2]]
    # points = points.cuda()
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)
    draw_point_cloud(points[0])

    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)
    print(global_feat)

    draw_point_cloud(torch.transpose(reconstructed_points,1,2)[0])


test_shape(5)












#####################################
# Test Chamfer Distance #############
#####################################

# Just for testing, not important






"""

def chamfer_distancenp(x, y, metric='l2', direction='bi'):
    Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}

    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

points = []
for _ in range(2):
    points.append(np.array(shape_maker1(3,400)).T)
# points = points.cuda()
points = np.array(points)
x = points[0].T
draw_point_cloud(Tensor(x))
points = Variable( Tensor(points) , requires_grad=True).to(device)


reconstructed_points, global_feat = autoencoder(points)
print(reconstructed_points)

y = reconstructed_points.detach().numpy()[0].T
draw_point_cloud(Tensor(y))
dist, normals = chamfer_distance(points, reconstructed_points)
print(chamfer_distancenp())
train_loss = torch.mean(dist)

# Calculate the gradients using Back Propogation
print(train_loss)

t1 = np.array([[1.,0.],[2.,0.]])
t2 = np.array([[1.,1.],[2.,2.]])
print(chamfer_distancenp(t1,t2))
print(chamfer_distance(Tensor(np.array([t1])),Tensor(np.array([t2]))))
print(chamfer_distanceown(Tensor(np.array([t1])),Tensor(np.array([t2]))))
print(2.0+1.0/np.sqrt(2))


t1 = np.array([[ 0.1527,  0.3000,  0.1730],
         [ 0.2264,  0.3000, -0.3000]])
t2 = np.array([[ 0.0891,  0.1055, -0.0898],
         [ 0.0057, -0.0091, -0.0028]])
print(chamfer_distancenp(t1,t2))
print(chamfer_distance(Tensor(np.array([t1])),Tensor(np.array([t2]))))
print(chamfer_distanceown(Tensor(np.array([t1])),Tensor(np.array([t2]))))
print(2.0+1.0/np.sqrt(2))


t1 = Tensor([[[-0.0738,  0.0724,  0.2252],
         [ 0.2626, -0.3000,  0.3000]],

        [[-0.0300,  0.3000,  0.1154],
         [ 0.2123, -0.3000,  0.1615]]])
t2 = Tensor([[[-0.3406,  0.0560,  0.1559],
         [-0.2350, -0.0072, -0.0292]],

        [[-0.3602,  0.0558,  0.1431],
         [-0.2588,  0.0094, -0.0351]]])

print(chamfer_distanceown(t1, t2))

"""
