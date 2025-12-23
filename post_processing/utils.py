import numpy as np
import qutip as qt
from thewalrus.decompositions import williamson, blochmessiah

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
    Db, S = williamson(cov)
    
    # Extract symplectic eigenvalues from diagonal matrix
    nu = np.diag(Db)[::2]  # Take indices [0, 2] -> [nu1, nu2]
    return S, nu
    

# -------------------------
# Gaussian state -> Fock density matrix
# -------------------------
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

# -------------------------
# Build Gaussian unitary (approx) from symplectic matrix
# -------------------------
def gaussian_unitary_from_symplectic(S: np.ndarray, ncut: int = 20):
    """
    Build Gaussian unitary from symplectic matrix S using Bloch-Messiah decomposition.
    S = O1 @ diag(exp(r1), exp(r1), exp(r2), exp(r2)) @ O2
    """
    # Bloch-Messiah decomposition
    O1, D, O2 = blochmessiah(S)
    
    # Extract diagonal elements
    diag_D = np.diag(D)  # [d1, d2, d1^-1, d2^-1]
    
    # Squeezing parameters: r such that d = exp(r)
    # For vacuum/no squeezing: d = 1, r = 0
    d1 = diag_D[0]
    d2 = diag_D[1]
    
    r1 = np.log(d1) if d1 > 0 else 0.0
    r2 = np.log(d2) if d2 > 0 else 0.0

    # Build bosonic operators
    a1 = qt.tensor(qt.destroy(ncut), qt.qeye(ncut))
    a2 = qt.tensor(qt.qeye(ncut), qt.destroy(ncut))
    
    # Build unitaries: O1, squeeze(r1), squeeze(r2), O2
    U_O1 = _symplectic_to_unitary(O1, ncut)
    U_O2 = _symplectic_to_unitary(O2, ncut)
    
    # Squeeze unitaries (handle zero squeezing)
    if abs(r1) > 1e-10:
        U_S1 = (r1 * (a1.dag()**2 - a1**2) / 2.0).expm()
    else:
        U_S1 = qt.qeye([ncut, ncut])
    
    if abs(r2) > 1e-10:
        U_S2 = (r2 * (a2.dag()**2 - a2**2) / 2.0).expm()
    else:
        U_S2 = qt.qeye([ncut, ncut])
    
    # Compose: O2 @ diag(squeeze) @ O1
    U = U_O2 * (U_S1 * U_S2) * U_O1
    
    return U

def _symplectic_to_unitary(S_ortho: np.ndarray, ncut: int):
    """Convert orthogonal symplectic to qutip unitary (beamsplitter)."""
    # For 2-mode: extract beamsplitter angle from S_ortho
    # S_ortho rotation ~ exp(i*theta*(a1^† a2 - a1 a2^†))
    theta = np.arctan2(S_ortho[0, 2], S_ortho[0, 0]) 
    a1 = qt.tensor(qt.destroy(ncut), qt.qeye(ncut))
    a2 = qt.tensor(qt.qeye(ncut), qt.destroy(ncut))
    return (theta * (a1.dag() * a2 - a1 * a2.dag())).expm()

def von_neumann_entropy(rho: qt.Qobj):
    evals = rho.eigenenergies()
    evals = np.real_if_close(evals)
    evals = np.maximum(evals, 0.0)
    tot = np.sum(evals)
    if tot <= 0:
        return 0.0
    probs = evals / tot
    probs = probs[probs > 0]
    S = -np.sum(probs * np.log2(probs))
    return S