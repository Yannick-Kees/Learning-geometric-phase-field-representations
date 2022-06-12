from loss_functionals import *


def kappa(f,g, n, d):
    
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    Fgradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    Ggradients = gradient(start_points,g(start_points))             # Calculate their gradients [ Dx_i ]
    Fnorms = Fgradients.norm(2,dim=-1)                            # [ |Dx_i| ]
    Gnorms = Ggradients.norm(2,dim=-1)                            # [ |Dx_i| ]
    kappaG = torch.trace( (1.0/(Gnorms))* torch.autograd.functional.hessian(g, start_points)  )
    kappaF = torch.trace((1.0/(Fnorms))* torch.autograd.functional.hessian(f, start_points)  )
    loss = (kappaG - kappaF)**2
    
    return loss.mean()
    




def sharpening(G, F, lambda_1, lambda_2, n, d, eps):
    
    
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible


    return torch.abs(F(start_points)-G(start_points)).mean()+ lambda_1 *  AT_Phasefield(G, eps, n, d)+lambda_2 * kappa(F,G,n,d)
    