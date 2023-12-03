import torch
import numpy as np

def gaussian_2d_batch(mean,covariance,image_size):
    """
    Function to generate batch of 2D Gaussian distributions
    given batch of mean and covariance matrix
    Given the mean and covariance matrix of a 2D Gaussian distribution, the following function
    ensures that the covariance matrix is positive definite, by using eigenvalue decomposition
    to get the eigenvalues less that zero and then adding a small value to the diagonal of the
    covariance matrix to make it positive definite.
    Paramters
    --------
    mean : tensor (batch_size,2)
    covariance : tensor (batch_size,2,2)
    image_size : tuple (width,height)
    Returns
    -------
    gaussian : tensor (batch_size,width,height)

    """
    width , height, _  = image_size
    #create a grid of points to sample the gaussian at
    #If the grid size is equal to the image size generate x , y for then image
    x, y = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height))
    pos = torch.dstack((x, y)).repeat(mean.shape[0],1,1,1).to(mean.device)
    #decompose covariance matrix to get eigenvalues
    L,V = torch.linalg.eig(covariance)
    #get the minimum eigenvalues
    min_eigenvalue = torch.min(torch.real(L),dim=1).values
    # get inxdex of eigenvalues less than zero
    idx = torch.nonzero(min_eigenvalue < 0)
    # if there are any negative eigenvalues then add a small value to the diagonal of the covariance matrix
    if len(idx) > 0:
        #update the covariance matrix by adding a small value to the diagonal
        update = 10 * min_eigenvalue[idx].view(*min_eigenvalue[idx].shape,1) * \
        torch.eye(*covariance[idx][0].shape[1:]).repeat(covariance[idx].shape[0],1,1).to(mean.device)
        #since here covariance is leaf node, which PyTorch does not allow in-place operation
        #we need to use torch.no_grad() to avoid the error
        with torch.no_grad():
            covariance[idx] -= update.unsqueeze(1)
    
    # Calculate the multivariate normal distribution
    #calculate the inverse of the covariance matrix and the determinant
    inv_covariance = torch.inverse(covariance)
    det_covariance = torch.det(covariance)
    # if the determinat contains a negative value then the matrix is not positive definite
    # and the gaussian is not defined, since the sqrt will yield a nan value.
    # To avoid this we add a small value to the determinant, or we can create it a lower triangular matrix
    constant = 1 / (2 * np.pi * torch.sqrt(det_covariance))
    exponent = -0.5 * torch.einsum('k...l,klm,k...m->k...', pos - mean.view(mean.shape[0],1,1,2), inv_covariance, pos - mean.view(mean.shape[0],1,1,2))
    gaussian = constant.view(-1, 1, 1) * torch.exp(exponent)
    return gaussian


def gaussian_2d_batch_unstable(mean,covariance,image_size):
    """
    Function to generate batch of 2D Gaussian distributions
    given batch of mean and covariance matrix
    Given the mean and covariance matrix of a 2D Gaussian distribution, the following function
    ensures that the covariance matrix is positive definite, by using eigenvalue decomposition
    to get the eigenvalues less that zero and then adding a small value to the diagonal of the
    covariance matrix to make it positive definite.
    Paramters
    --------
    mean : tensor (batch_size,2)
    covariance : tensor (batch_size,2,2)
    image_size : tuple (width,height)
    Returns
    -------
    gaussian : tensor (batch_size,width,height)

    """
    width , height, _  = image_size
    #create a grid of points to sample the gaussian at
    #If the grid size is equal to the image size generate x , y for then image
    x, y = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height))
    pos = torch.dstack((x, y)).repeat(mean.shape[0],1,1,1).to(mean.device)
    #decompose covariance matrix to get eigenvalues
    L,V = torch.linalg.eig(covariance)
    #get the minimum eigenvalues
    min_eigenvalue = torch.min(torch.real(L),dim=1).values
    # get inxdex of eigenvalues less than zero
    idx = torch.nonzero(min_eigenvalue < 0)
    # if there are any negative eigenvalues then add a small value to the diagonal of the covariance matrix
    if len(idx) > 0:
        #update the covariance matrix by adding a small value to the diagonal
        update = 10 * min_eigenvalue[idx].view(*min_eigenvalue[idx].shape,1) * \
        torch.eye(*covariance[idx][0].shape[1:]).repeat(covariance[idx].shape[0],1,1).to(mean.device)
        #Since covariance is no longer a leaf node we can use in-place operation
        covariance[idx] -= update.unsqueeze(1)
    
    # Calculate the multivariate normal distribution
    #calculate the inverse of the covariance matrix and the determinant
    inv_covariance = torch.inverse(covariance)
    det_covariance = torch.det(covariance)
    # if the determinat contains a negative value then the matrix is not positive definite
    # and the gaussian is not defined, since the sqrt will yield a nan value.
    # To avoid this we add a small value to the determinant, or we can create it a lower triangular matrix
    constant = 1 / (2 * np.pi * torch.sqrt(det_covariance))
    exponent = -0.5 * torch.einsum('k...l,klm,k...m->k...', pos - mean.view(mean.shape[0],1,1,2), inv_covariance, pos - mean.view(mean.shape[0],1,1,2))
    gaussian = constant.view(-1, 1, 1) * torch.exp(exponent)
    return gaussian