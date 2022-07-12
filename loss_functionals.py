from networks import *

def gradient(inputs, outputs):
    # Returns:
    #   Pointwise gradient estimation [ Df(x_i) ]
    
    # Parameters:
    #   inputs:     [ x_i ]
    #   outputs:    [ f(x_i) ] 
    
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True)[0][:,-3:]
    return points_grad

#############################
# PHASE - Loss ##############
#############################

# double well potential
#W = lambda s: s**2 - 2.0*torch.abs(s) + torch.tensor([1.0]).to(device) 
W = lambda s: (9.0/16.0) *  (s**2 -torch.tensor([1.0]).to(device)   )**2


def ModicaMortola(f, eps, n, d):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return (W(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2


def Zero_recontruction_loss_Lip(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    
    n = len(pc)
    
    matrix = pc.repeat(m,1).to(device) 
    matrix = torch.reshape(matrix, (m,n,d)) # 3D Matrix containing the points
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) ).to(device) 
    error = torch.reshape( variation, (m,n, d) ) # 3D Matrix containing normal distribution
    matrix += error
    matrix = matrix.reshape(m*n,d)
    matrix = f(matrix)  # Apply network to targets
    matrix = torch.reshape( matrix, (m,n) ).mean(0)
    matrix = torch.abs(matrix).mean()

    return  c*eps**(1.0/3.0) *  matrix      # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} u(x) dx|


def Eikonal_loss(f, pc, eps, d):
    # Returns:
    #   Eikonal loss around the points of point cloud
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    
    gradients = gradient( pc, f(pc) )                                               # calculates [ Du(x) ]
    norms = np.sqrt(eps) *  gradients.norm(2,dim=-1)                                # calculates [ |Dw(x)| ] = [ sqrt(eps) * |Du(x)| ]
    eikonal = torch.abs(torch.full(size=(len(pc) ,1), fill_value=1.0).to(device)-norms.to(device))**2     # calculates [ | 1-|Dw(x)| |^2 ] 
    
    return eikonal.mean()    # return \sum_{x\in X} |1-|Dw(u) | |^2                 # returns [ | 1-|Dw(x)| |^2 ] 

def Phase_loss(f, pointcloud, eps, n, m, c, mu):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    #   mu:     Constant \mu, contribution of Eikonal equation
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return eps**(-.5)*(ModicaMortola(f, eps, n, d) +  Zero_recontruction_loss_Lip(f, pointcloud, eps, m, c, d))+mu * Eikonal_loss(f, pointcloud, eps, d )


def test_MM_GV(f, pc, eps, n, m, c, p):
    d = pc.shape[1]
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    
    (W(f(start_points))+eps*norms).mean()   
    EINS = 1.0/(eps) * W(f(start_points)).mean()
    ZWEI  = (eps * norms).mean()
    n = len(pc)
    
    matrix = pc.repeat(m,1).to(device) 
    matrix = torch.reshape(matrix, (m,n,d)) # 3D Matrix containing the points
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) ).to(device) 
    error = torch.reshape( variation, (m,n, d) ) # 3D Matrix containing normal distribution
    matrix += error
    matrix = matrix.reshape(m*n,d)
    matrix = f(matrix)  # Apply network to targets
    matrix = torch.reshape( matrix, (m,n) ).mean(0)
    DREI = c * eps**(-1.0/3.0) * torch.abs(matrix).mean()
    if p:
        print("1: ",EINS,"2: ",ZWEI,"3: ",DREI)
    return  EINS+ZWEI+DREI


#############################
# Ambrosio Tortorelli #######
#############################

# One well potential
U = lambda s: (s- torch.tensor([1.0]).to(device))**2

# Shifting function
g = lambda s: s**2


def AT_Phasefield(f, eps, n, d):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return ( (1.0/(4*eps))  * U(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2



def Zero_recontruction_loss_AT(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    

    return  c*eps**(-1.0/3.0) *  ( torch.abs(f(pc)).mean() )            # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|


def Zero_recontruction_loss_AT_Shift(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    
    n = len(pc)
    
    matrix = pc.repeat(m,1)
    matrix = torch.reshape(matrix, (m,n,d))         # 3D Matrix containing the points
    variation  = torch.normal(mean = torch.full(size=( n*m *d,1), fill_value=0.0) , std= torch.full(size=(m*n*d,1), fill_value=.001) )
    error = torch.reshape( variation, (m,n, d) )    # 3D Matrix containing normal distribution
    matrix += error
    matrix = matrix.reshape(m*n,d)
    matrix = g(f(matrix))                           # Apply network to targets and shift values
    matrix = torch.reshape( matrix, (m,n) ).mean(0)
    matrix = torch.abs(matrix).mean()

    return  c*eps**(-1.0/3.0) *  matrix              # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|



def AT_loss(f, pointcloud, eps, n, m, c):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    #   mu:     Constant \mu, contribution of Eikonal equation
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return AT_Phasefield(f, eps, n, d) +  Zero_recontruction_loss_AT(f, pointcloud, eps, m, c, d)



#############################
# Loss on L^2 ###############
#############################


def L2_Loss(f, input, Batch):
    
    # Returns:
    #   Integral over manifold \int_S f(x) dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   input:  Points on the manifold
    #   Batch:  Number of integral evaluations

    indices = np.random.choice(len(input), Batch, False)
    x = Variable( Tensor(input[indices])).to(device)

    return f(x).mean()







#####################################
# Shapespace Learning ###############
#####################################




def AT_Phasefield_shapespace(f, eps, n, d, fv):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5).to(device)    # Create random points [ x_i ]
    features = fv.repeat(n,1)
    start_points = torch.cat((start_points, features), 1)
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return ( (1.0/(4*eps))  * U(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2



def Zero_recontruction_loss_AT_shapespace(f, pc, eps, c, fv):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    
    features = fv.repeat(400,1)    
    pc = torch.cat((pc, features), 1)

    return  c*eps**(-1.0/3.0) *  ( torch.abs(f(pc)).mean() )            # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|


def AT_loss_shapespace(f, pointcloud, eps, n, c, fv):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    #   mu:     Constant \mu, contribution of Eikonal equation
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return AT_Phasefield_shapespace(f, eps, n, d, fv) +  Zero_recontruction_loss_AT_shapespace(f, pointcloud, eps, c, fv)








def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distanceown(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
    Returns:
        2-element tuple containing
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    

    cham_dist = cham_x + cham_y
    

    return cham_dist