from mpi4py import MPI
from scipy import linalg

from lib import *

def posterior_approx_distributed(comm, dim_param, MPIROOT=0):
    '''
    Compute normal approximation to a posterior distribution based upon normal
    approximations to the posterior computed on each worker. Collects
    information matrices and information-weighted parameter estimates from
    workers, then uses these to construct (via Fisher weighting) a new proposal
    distribution.

    Parameters
    ----------
        - comm : MPI communicator
            Communicator from which to collect normal approximations
        - dim_param : int
            Size of parameter.
        - MPIROOT : int
            Rank of root for communicator. Defaults to 0.

    Returns
    -------
        - est : array_like
            1d array of length dim_param containing approximate posterior mean
            for parameter.
        - prec : array_like
            2d array of shape (dim_param, dim_param) containing approximate
            posterior precision for parameter.
    '''
    # Using simple 1d format to send point estimates and precisions together.
    # Define dim_prec as dim_param*(dim_param+1)/2:
    #   - 0:dim_param : point estimate
    #   - dim_param:(dim_prec + dim_param) : lower-triangular portion of info
    dim_prec = (dim_param * (dim_param + 1)) / 2
    buf = np.zeros(dim_param + dim_prec, dtype=np.float)
    approx = np.zeros(dim_param + dim_prec, dtype=np.float)

    # Compute sum of all point estimates and precisions
    comm.Reduce([buf, MPI.DOUBLE], [approx, MPI.DOUBLE],
                op=MPI.SUM, root=MPIROOT)

    # Extract precision matrix
    prec = np.empty((dim_param, dim_param))
    ind_l = np.tril_indices(dim_param)
    prec[ind_l] = approx[dim_param:]
    prec.T[ind_l] = prec[ind_l]

    # Compute approximate posterior mean from information-weighted estimates
    est = approx[:dim_param]
    est = linalg.solve(prec, est, sym_pos=True, lower=True)

    return (est, prec)


def refine_distributed_approx(comm, est, prec, dim_param, n_iter=1,
                              final_info=1, MPIROOT=0):
    '''
    Execute single distributed Newton-Raphson step starting from
    precision-weighted approximation. This refines the approximation; inspired
    by 1-step efficient estimators.

    Parameters
    ----------
        - comm : MPI communicator
            Communicator from which to collect normal approximations
        - dim_param : int
            Size of parameter.
        - MPIROOT : int
            Rank of root for communicator. Defaults to 0.
        - est : array_like
            1d array of length dim_param containing approximate posterior mean
            for parameter, as output by posterior_approx_distributed.
        - prec : array_like
            2d array of shape (dim_param, dim_param) containing approximate
            posterior precision for parameter, as output by
            posterior_approx_distributed.

    Returns
    -------
        - est : array_like
            1d array of length dim_param containing approximate posterior mean
            for parameter, refined via distributed NR step.
        - prec : array_like
            2d array of shape (dim_param, dim_param) containing approximate
            posterior precision for parameter, refined via distributed NR step.
    '''
    # Broadcast number of iterations and final information update flag to
    # workers
    settings = np.array([n_iter, final_info], dtype=int)
    comm.Bcast([settings, MPI.INT], root=MPIROOT)

    # Initialize buffers
    # Using simple 1d format to send gradients and negative Hessians together.
    # Define dim_hess as dim_param*(dim_param+1)/2:
    #   - 0:dim_param : gradient
    #   - dim_param:(dim_hess + dim_param) : lower-triangular portion of Hessian
    dim_hess = (dim_param * (dim_param + 1)) / 2
    buf = np.zeros(dim_param + dim_hess, dtype=np.float)
    update = np.zeros(dim_param + dim_hess, dtype=np.float)
    hess = np.empty((dim_param, dim_param))
    ind_l = np.tril_indices(dim_param)

    for i in xrange(n_iter):
        # Broadcast current estimate to workers
        comm.Bcast([est, MPI.DOUBLE], root=MPIROOT)

        # Compute sum of all gradients and Hessians
        buf[:] = 0.
        comm.Reduce([buf, MPI.DOUBLE], [update, MPI.DOUBLE],
                    op=MPI.SUM, root=MPIROOT)

        # Extract negative Hessian matrix
        hess[ind_l] = update[dim_param:]
        hess.T[ind_l] = hess[ind_l]

        # Update approximation with single Newton-Raphson step
        est += linalg.solve(hess, update[:dim_param])

    if final_info:
        # Broadcast updated point estimate
        comm.Bcast([est, MPI.DOUBLE], root=MPIROOT)

        # Collect updated Hessian
        comm.Reduce([buf[dim_param:], MPI.DOUBLE],
                    [update[dim_param:], MPI.DOUBLE], op=MPI.SUM,
                    root=MPIROOT)

        # Extract updated negative Hessian matrix
        hess = np.empty((dim_param, dim_param))
        ind_l = np.tril_indices(dim_param)
        hess[ind_l] = update[dim_param:]
        hess.T[ind_l] = hess[ind_l]

    return (est, hess)

