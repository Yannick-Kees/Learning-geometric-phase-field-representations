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
W = lambda s: s**2 - 2.0*torch.abs(s) + torch.tensor([1.0])
#W = lambda s: (s- torch.tensor([1.0]))**2

def ModicaMortola(f, eps, n, d):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)  # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return (W(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2


def Zero_recontruction_loss(f, pc, eps, n, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    
    loss = 0    # loss in the sum
    
    for point in pc:
        # loop over all points in pointcloud, i.e. x\in X
        
        variation  = torch.normal(mean = torch.full(size=( n* d,1), fill_value=0.0) , std= torch.full(size=(n*d,1), fill_value=.001) )  # Random points in B_\delta(0)
        start_point = point.repeat(n,1)+   torch.reshape( variation, (n, d) )                                                             # Random points [ x_i ] in B_\delta(x)
        start_point = start_point.to(device)
        loss +=  torch.abs(f(start_point).mean())                                                                                           # Estimate \sum_{x\in X} |\dashint_{B_delta} u(x) dx|

    return  c*eps**(1.0/3.0)/(len(pc)) *  loss      # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} u(x) dx|


def Eikonal_loss(f, pc, eps, d):
    # Returns:
    #   Eikonal loss around the points of point cloud
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    
    gradients = gradient( pc, f(pc) )                                               # calculates [ Du(x) ]
    norms = np.sqrt(eps) *  gradients.norm(2,dim=-1)                                # calculates [ |Dw(x)| ] = [ sqrt(eps) * |Du(x)| ]
    eikonal = torch.abs(torch.full(size=(len(pc) ,1), fill_value=1.0)-norms)**2     # calculates [ | 1-|Dw(x)| |^2 ] 
    
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
    
    return eps**(-.5)*(ModicaMortola(f, eps, n, d) +  Zero_recontruction_loss(f, pointcloud, eps, m, c, d))+mu * Eikonal_loss(f, pointcloud, eps, d )


#############################
# Ambrosio Tortorelli #######
#############################


def AT_loss(f, pointcloud, eps, n, m, c, mu):
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
    
    return eps**(-.5)*(ModicaMortola(f, eps, n, d) +  Zero_recontruction_loss(f, pointcloud, eps, m, c, d))+mu * Eikonal_loss(f, pointcloud, eps, d )



