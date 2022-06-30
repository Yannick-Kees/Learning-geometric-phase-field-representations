from shapemaker import *

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

NUM_TRAINING_SESSIONS = 200
num_points = 400
Batch_size = 20

autoencoder = PCAutoEncoder2(3, num_points)
autoencoder.to(device)


optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


for epoch in range(NUM_TRAINING_SESSIONS+1):

    points = []
    for _ in range(Batch_size):
        points.append(shape_maker1(3,num_points))
    # points = points.cuda()
    points = np.array(points)
    points = Variable( Tensor(points) , requires_grad=True).to(device)

    optimizer.zero_grad()
    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)

    dist = chamfer_distanceown(points, torch.transpose(reconstructed_points, 1, 2))


    train_loss = torch.mean(dist)

    # Calculate the gradients using Back Propogation
    train_loss.backward() 

    # Update the weights and biases 
    optimizer.step()
    report_progress(epoch, NUM_TRAINING_SESSIONS , train_loss.detach().cpu().numpy() )
    if epoch % 50 == 0:
        torch.save(autoencoder.state_dict(), 'autoencoder.pth')
        

    scheduler.step()

torch.save(autoencoder.state_dict(), 'autoencoder2.pth')

print("Finished")