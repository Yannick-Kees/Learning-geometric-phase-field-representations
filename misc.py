from packages import *

def report_progress(current, total, error):
   # Prints out, how far the training process is

   # Parameters:
   #     current:    where we are right now
   #     total:      how much to go
   #     error:      Current Error, i.e. evaluation of the loss functional
   
   sys.stdout.write('\rProgress: {:.2%}, Current Error: {:}'.format(float(current)/total, error))
   if current==total:
      sys.stdout.write('\n')
   sys.stdout.flush()
   

   
   
# enable computing on GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_off(file):
   # returning matrix of points from an .off file

   # Parameters:
   #   file:   path of file to read
   
   if 'OFF' != file.readline().strip():
      # Not an .off file    
      raise('Not a valid OFF header')
   
   n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
   verts = [[float(s) for s in file.readline().strip().split()] for _ in range(n_verts)]
   return verts



def read_obj(file):
   # returning matrix of points from an .obj file

   # Parameters:
   #   file:   path of file to read
   
   if 'OFF' != file.readline().strip():
      # Not an .off file    
      raise('Not a valid OFF header')
   
   n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
   verts = [[float(s) for s in file.readline().strip().split()] for _ in range(n_verts)]
   return verts

def read_ply_file(file):
       
   # returning matrix of points from an .ply file
   # needed for large numbers of points

   # Parameters:
   #   file:   path of file to read
    
    data = file.readlines()
    num_vertices = int(data[4].split(" ")[2].replace("\n","")) # Number of vertices are stored in line 

    vertices = [   [float(x)  for x in row.split(" ")[0:3] ] for row in data[11:11+num_vertices]]
    return vertices


def read_obj_file(file):
      pc = []
      for line in file.readlines():
         x = line.replace("\n","").split(" ")
         if x[0] != "v":
            break
         else:
            pc.append(   [float(xx)  for xx in x[1:4] ]   )
      return pc      
                  


def CreateFourierMatrix(size, d):
   print([[ np.random.normal()  for _ in range(d)] for _ in range(size)])
   
   
 