import numpy as np
import qutip as qt
from thewalrus.decompositions import williamson, blochmessiah, sympmat
from scipy.linalg import logm

def normalize(alice_symbols, bob_symbols, shot, photon_number):
    """ Normalize Alice and Bob symbols."""
    dataAlice_normalized = alice_symbols * np.sqrt(photon_number / np.mean(np.abs(alice_symbols)**2))
    dataBob_normalized = bob_symbols / np.sqrt(shot)

    return dataAlice_normalized, dataBob_normalized

def williamson_decomposition(cov: np.ndarray):
    """
    Numerical Williamson decomposition for 2-mode covariance matrix.
    Returns: S (symplectic matrix 4x4), nu (array [nu1, nu2])
    such that: cov = S.T @ diag([nu1,nu1,nu2,nu2]) @ S
    Reference: numerical recipe using diagonalization of i Omega sigma.
    """
    Db, S= williamson(cov)
    
    # Extract symplectic eigenvalues from diagonal matrix
    nu = np.diag(Db)[::2]  # Take indices [0, 2] -> [nu1, nu2]
    return S, nu
    
def homodyne_gaussian_to_densitymatrix(cov: np.ndarray, mu: np.ndarray, ncut: int = 20):
    """
    Convert a 2-mode Gaussian state to qutip density matrix.
    cov: 4x4 covariance in (q,p,q,p) ordering
    mu: length-4 vector (q1,p1,q2,p2)
    """
    S, nu = williamson_decomposition(cov)
    
    # Thermal state with Williamson eigenvalues
    nbar = np.maximum((nu - 1.0) / 2.0, 0.0)
    rho_th = qt.tensor(qt.thermal_dm(ncut, nbar[0]), 
                       qt.thermal_dm(ncut, nbar[1]))
    
    # Apply Bloch-Messiah unitary
    U_G = gaussian_unitary_from_symplectic(S, ncut)
    rho = U_G * rho_th * U_G.dag()
    
    alpha1 = (mu[0]) / 2.0
    alpha2 = (mu[2]) / 2.0
    D = qt.tensor(qt.displace(ncut, alpha1), qt.displace(ncut, alpha2))
    rho_final = D * rho * D.dag()
    
    return rho_final

def heterodyne_gaussian_to_densitymatrix(cov: np.ndarray, mu: np.ndarray, ncut: int = 20):
    """
    Convert a 2-mode Gaussian state to qutip density matrix.
    cov: 4x4 covariance in (q,p,q,p) ordering
    mu: length-4 vector (q1,p1,q2,p2)
    """
    S, nu = williamson_decomposition(cov)
    
    # Thermal state with Williamson eigenvalues
    nbar = np.maximum((nu - 1.0) / 2.0, 0.0)
    rho_th = qt.tensor(qt.thermal_dm(ncut, nbar[0]), 
                       qt.thermal_dm(ncut, nbar[1]))
    
    # Apply Bloch-Messiah unitary
    U_G = gaussian_unitary_from_symplectic(S, ncut)
    rho = U_G * rho_th * U_G.dag()
    
    alpha1 = (mu[0] + 1j * mu[1]) / 2.0
    alpha2 = (mu[2] + 1j * mu[3]) / 2.0
    D = qt.tensor(qt.displace(ncut, alpha1), qt.displace(ncut, alpha2))
    rho_final = D * rho * D.dag()
    
    return rho_final

def gaussian_unitary_from_symplectic(S: np.ndarray, ncut: int = 20):
    """
    Builds the QuTiP unitary operator U corresponding to symplectic matrix S
    using the Bloch-Messiah decomposition. 
    S = O2 @ D @ O1  ->  U = U_O2 * U_sq * U_O1
    """
    # 1. Bloch-Messiah decomposition (S = O1 @ D @ O2 in thewalrus notation)
    O1, D, O2 = blochmessiah(S)
    
    # 2. Reconstruct Passive Unitaries
    # Note: O1 and O2 are applied first/last depending on vector convention.
    # We maintain the order U = U_O1 * U_sq * U_O2 matching S operator order.
    U_O1 = _passive_symplectic_to_unitary(O1, ncut)
    U_O2 = _passive_symplectic_to_unitary(O2, ncut)
    
    # 3. Reconstruct Squeezing Unitary
    N = S.shape[0] // 2
    diag_D = np.diag(D)
    
    op_list = []
    for i in range(N):
        # FIX: Access the q-quadrature scaling for mode i.
        # D is ordered (q1, p1, q2, p2, ...).
        # Mode i's q-scaling is at index 2*i.
        d_val = diag_D[2*i]
        
        # Calculate squeezing parameter r
        # Scaling q -> d * q corresponds to squeeze factor r = -log(d)
        r = -np.log(d_val)
        
        if abs(r) > 1e-10:
            op_list.append(qt.squeeze(ncut, r))
        else:
            op_list.append(qt.qeye(ncut))
            
    if N == 1:
        U_sq = op_list[0]
    else:
        U_sq = qt.tensor(op_list)

    # 4. Compose Total Unitary
    U = U_O1 * U_sq * U_O2
    return U

def _passive_symplectic_to_unitary(O: np.ndarray, ncut: int):
    """
    Converts an Orthogonal Symplectic matrix O (passive transform) to a QuTiP Unitary.
    Corrects for sign conventions and Identity matrix edge cases.
    """
    N = O.shape[0] // 2
    
    # 1. Reorder from (q1, p1, ...) to (q1...qN, p1...pN) to extract X, Y blocks
    perm_inds = []
    for i in range(N): perm_inds.append(2*i)     # q indices
    for i in range(N): perm_inds.append(2*i+1)   # p indices
    
    O_block = O[perm_inds][:, perm_inds]
    
    # Extract blocks: S_passive = [[X, Y], [-Y, X]]
    X = O_block[:N, :N]
    Y = O_block[:N, N:]
    
    # 2. Construct Complex Unitary
    # Use (X - iY) to match annihilation operator transformation a' = U a
    U_c = X - 1j * Y
    
    # 3. Compute Hamiltonian Generator
    # U_c = exp(-i H_mat)  =>  H_mat = i * logm(U_c)
    H_mat = 1j * logm(U_c)
    
    # 4. Lift to Fock Space Operator
    # Use a Qobj of zeros to avoid 'complex' scalar errors
    # if H_mat is all zeros (Identity case).
    
    # Create a list of Identity operators for tensor product structure
    id_list = [qt.qeye(ncut) for _ in range(N)]
    if N == 1:
        H_op = id_list[0] * 0.0
    else:
        H_op = qt.tensor(id_list) * 0.0
        
    # Build annihilation operators list for the Hamiltonian sum
    a_ops = []
    for i in range(N):
        ops = [qt.qeye(ncut) for _ in range(N)]
        ops[i] = qt.destroy(ncut)
        if N == 1:
            a_ops.append(ops[0])
        else:
            a_ops.append(qt.tensor(ops))
        
    # Sum terms: H_op = sum H_jk * a_j^dag * a_k
    for j in range(N):
        for k in range(N):
            coeff = H_mat[j, k]
            if abs(coeff) > 1e-10:
                H_op += coeff * a_ops[j].dag() * a_ops[k]
                
    # 5. Exponentiate to get Unitary
    return (-1j * H_op).expm()
