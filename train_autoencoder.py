from shapemaker import *

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

NUM_TRAINING_SESSIONS = 5000
num_points = 1000
Batch_size = 40

autoencoder = PCAutoEncoder64(3, num_points)
autoencoder.to(device)

dataset = np.load(open("dataset1k.npy", "rb"))


optimizer = optim.Adam(autoencoder.parameters(), lr=0.001 )
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=40, verbose=False)


for epoch in range(NUM_TRAINING_SESSIONS+1):

    indices = np.random.choice(len(dataset), Batch_size, False)
    pointcloud = dataset[indices]#[:,:num_points]
    points = Variable( Tensor(pointcloud) , requires_grad=False).to(device)

    autoencoder.zero_grad()
    inputs = torch.transpose(points, 1, 2)
    reconstructed_points, global_feat = autoencoder(inputs)

    dist = chamfer_distance(points, torch.transpose(reconstructed_points, 1, 2))
    # dist = torch.mean(torch.abs(points - torch.transpose(reconstructed_points, 1, 2)),2)
    
    train_loss = dist[0]

    # Calculate the gradients using Back Propogation
    train_loss.backward() 

    # Update the weights and biases 
    optimizer.step()
    report_progress(epoch, NUM_TRAINING_SESSIONS , train_loss.detach().cpu().numpy() )
        

    scheduler.step(train_loss)

torch.save(autoencoder.state_dict(), 'autoencoder64.pth')

print("Finished")