#from chamfer_distance import ChamferDistance
from shapemaker import *
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

NUM_TRAINING_SESSIONS = 1
num_points = 10

autoencoder = PCAutoEncoder(3, num_points)


optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


for epoch in range(NUM_TRAINING_SESSIONS+1):

    points = []
    for _ in range(2):
        points.append(np.array(shape_maker1(3,num_points)).T)
    # points = points.cuda()
    points = Variable( Tensor(points) , requires_grad=True).to(device)
    print(points)
    print(points.size())
    optimizer.zero_grad()
    reconstructed_points, global_feat = autoencoder(points)

    dist1, dist2 = chamfer_distance(points, reconstructed_points)
    print(dist1)
    print(dist2)
    train_loss = torch.mean(dist1)

    # Calculate the gradients using Back Propogation
    train_loss.backward() 

    # Update the weights and biases 
    optimizer.step()
    report_progress(epoch, NUM_TRAINING_SESSIONS , train_loss.detach().cpu().numpy() )

    scheduler.step()
    #torch.save(autoencoder.state_dict(), 'saved_models/autoencoder_%d.pth' % (epoch))
        
