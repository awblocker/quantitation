import numpy as np
from scipy import linalg
from scipy import spatial

# Covariance functions

def cov_sqexp(r, scale=1.):
    '''
    Squared exponential (Gaussian) covariance function with given scale.
    '''
    return np.exp( - (r / scale)**2 )

def build_emulator(f, center, slope_mean=None, cov=cov_sqexp, grid_radius=1.,
                   grid_transform=None, grid_min_spacing=0.5,
                   grid_shape='spherical', f_args=(), f_kwargs={}, cov_args=(),
                   cov_kwargs={}, min_cov=1e-9, cov_step=10.,
                   max_log10_condition=10.):
    '''
    Build Gaussian processor emulator for given function.
    
    Parameters
    ---------- 
    f : function
        Function to emulate. Must take a d x n matrix as its first argument and
        return return a k x n ndarray containing the function value for
        each d-dimensional column of the input matrix. Called as
        f(x, *f_args, **f_kwargs) in evaluations.
    center : d length ndarray
        Center of sampling region for emulator.
    slope_mean : d x d ndarray
        Optional linear approximation for f(x - center).
    cov : function
        Covariance function for Gaussian process. Must accept ndarray of
        distances as first argument and return an ndarray of the same dimension.
        Called as cov(dm, *cov_args, **cov_kwargs).
    grid_radius : number
        Minimum radius of grid before transform, inclusive.
    grid_transform : np.ndarray or matrix
        Optional d x d nd.array or matrix providing transformation from cubic or
        spherical grid into space of interest. Should be lower-triangular and
        positive-definite.
    grid_min_spacing : float
        Minimum spacing of grid after transformation.
    grid_shape : string
        Shape of grid, 'cubic' or 'spherical'. Spherical is truncated cubic
        grid.
    f_args : tuple
        Tuple of additional positional arguments for f.
    f_kwargs : dict
        Dictionary of additional kw arguments for f.
    cov_args : tuple
        Tuple of additional positional arguments for cov.
    cov_kwargs : tuple
        Dictionary of additional kw arguments for cov.
    min_cov : float
        Initial minimum covariance; covariance matrix is truncated at this
        value.
    cov_step : float
        Multiplicative step for minimum covariance (upward) if covariance matrix
        is computationally singular.
    max_log10_condition : number
        Maximum log10 condition number to accept for covariance matrix.
        Truncation continues at min_cov * cov_step**k until this is satisfied.

    Returns
    -------
    A dictionary containing:
        - grid : d x n_grid ndarray
            The computed grid for approximation
        - v : n_grid length ndarray
            Vector for approximation
        - center : d length ndarray
            Center of emulation region
        - slope_mean : d x d ndarray
            Optional slope of linear mean function. Can be None.
    '''
    # Get dimensions
    d = np.size(center)

    # Find eigenvalues of transformation
    if grid_transform is None:
        transform_eigenvalues = np.ones(d)
    else:
        transform_eigenvalues = np.diag(grid_transform)

    # Build grid before rotation and scaling, adjusting spacing as needed
    grid_radius = float(grid_radius)
    h_grid = [grid_radius / np.ceil(grid_radius / grid_min_spacing * v) for v
              in transform_eigenvalues]
    dim_grid = [int(2 * grid_radius / h + 1) for h in h_grid]
    dim_grid_float = np.array(dim_grid, dtype=float)
    grid = np.mgrid[tuple(slice(0, l) for l in dim_grid)]
    grid = np.array([z.flatten() for z in grid], dtype=float).T
    grid /= (dim_grid_float - 1.) / 2.
    grid -= 1.

    # Truncate to sphere if requested
    if grid_shape[:5] == 'spher':
        grid = grid[np.sum(grid**2, 1) <= 1]

    # Rescale for radius
    grid *= grid_radius

    # Transform and recenter
    if grid_transform is not None:
        grid = np.dot(grid, grid_transform.T)

    grid += center

    # Evaluate function over grid
    f_values = f(grid.T, *f_args, **f_kwargs)
    
    if slope_mean is not None:
        f_values -= np.dot(slope_mean, (grid - center).T)

    # Compute covariance matrix for GP
    C = spatial.distance_matrix(grid, grid, p=2)
    C = cov(C, *cov_args, **cov_kwargs)
    
    # Truncate at minimum covariance
    C[C < min_cov] = 0.
    
    # Continue to truncate at higher covariances if needed for numerical
    # stability
    svals = linalg.svdvals(C)
    log10_condition = np.ptp(np.log10(svals))
    while log10_condition > max_log10_condition:
        min_cov *= cov_step
        C[C < min_cov] = 0.
        svals = linalg.svdvals(C)
        log10_condition = np.ptp(np.log10(svals))

    # Compute vector for subsequent approximations
    v = linalg.solve(C, f_values.T)

    # Build output
    emulator = {'grid' : grid, 'v' : v, 
                'center' : center, 'slope_mean' : slope_mean}

    return emulator

def evaluate_emulator(x, emulator, cov, cov_args=(), cov_kwargs={}):
    '''
    Evaluates emulator at given point or sequence of points
    
    Arguments
    ---------
    x : ndarray
        Array of length d or of dimension d x m, with each column containing a
        point at which to evaluate the emulator.
    emulator : dict
        Dictionary as output by build_emulator containing grid and v.
    cov : function
        Covariance function for Gaussian process. Must accept ndarray of
        distances as first argument and return an ndarray of the same dimension.
        Called as cov(dm, *cov_args, **cov_kwargs).
    cov_args : tuple
        Tuple of additional positional arguments for cov.
    cov_kwargs : tuple
        Dictionary of additional kw arguments for cov.
    
    Returns
    -------
    f_hat : ndarray
        Array of length m containing estimated values of function.
    '''
    # Convert x to matrix if needed
    if not type(x) is np.ndarray:
        x = np.array(x)
    if len(x.shape) < 2:
        x = x[:, np.newaxis]

    # Evaluate distances between x and grid
    C = spatial.distance_matrix(x.T, emulator['grid'])
    C = cov(C, *cov_args, **cov_kwargs)

    # Estimate function values at x
    f_hat = np.dot(C, emulator['v']).T
    
    # Add linear term if needed
    if emulator['slope_mean'] is not None:
        f_hat += np.dot(emulator['slope_mean'], (x.T - emulator['center']).T)

    return f_hat

