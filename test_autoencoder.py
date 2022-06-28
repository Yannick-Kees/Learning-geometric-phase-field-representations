from shapemaker import *
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

autoencoder = PCAutoEncoder(3, 400)
autoencoder.load_state_dict(torch.load(r"autoencoder.pth", map_location=device))




points = []
for _ in range(2):
    points.append(np.array(shape_maker1(3,400)).T)
# points = points.cuda()
points = np.array(points)
x = points[0].T
draw_point_cloud(Tensor(x))
points = Variable( Tensor(points) , requires_grad=True).to(device)


reconstructed_points, global_feat = autoencoder(points)
y = reconstructed_points.detach().numpy()[0].T
draw_point_cloud(Tensor(y))
dist, normals = chamfer_distance(points, reconstructed_points)

train_loss = torch.mean(dist)

# Calculate the gradients using Back Propogation
print(train_loss)



