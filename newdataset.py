from networks import *
npz_file = np.load("3dObjects\Mobius.npz")
position = npz_file['position']
distance = npz_file['distance']
gradient = npz_file['gradient']
print(distance.shape)
print(len(position))
indices = np.random.choice(len(position), 20000, False)
cloud = position[indices]
cloud = torch.tensor(cloud )

pc = Variable( cloud , requires_grad=True).to(device)

#print(pc)
#cloud = torch.tensor(flat_circle(8000) )

draw_point_cloud(pc)