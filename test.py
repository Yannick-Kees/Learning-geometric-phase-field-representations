from networks import *

file = open("3dObjects/cube.off")    
pc = read_off(file)
print(pc)
cloud = torch.tensor(normalize(pc) )
print(cloud)
