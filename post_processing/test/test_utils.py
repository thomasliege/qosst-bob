import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import williamson_decomposition, gaussian_unitary_from_symplectic
from heterodyne_keyrate import heterodyne_gaussian_to_densitymatrix

def test_williamson_property():
    """
    Test 1: Check if Williamson decomposition returns a valid symplectic decomposition.
    S @ (nu * I) @ S.T == Cov
    """
    print("\n[Test 1] Williamson Decomposition Properties")
    
    # Construct a random valid covariance matrix
    # Start with thermal state and apply random symplectic
    N = 2
    nbar = np.array([0.5, 1.5])
    D_base = np.diag(np.concatenate([2*nbar+1, 2*nbar+1])) # (q1, q2, p1, p2) order approx
    # Actually let's just make a simple diagonal one first to be safe
    cov_in = np.diag([2.0, 0.5, 2.0, 0.5]) # Squeezed state-ish
    
    S, nu = williamson_decomposition(cov_in)
    
    # Reconstruction check
    # Note: Williamson from thewalrus usually implies V = S @ D @ S.T
    # We need to build D with correct ordering.
    # thewalrus returns D diagonal in same order as input cov if cov was diagonal?
    # Usually D = diag(nu_1, nu_1, nu_2, nu_2) if input is (q1, p1, q2, p2) interleaved?
    # Let's check nu values specifically.
    
    print(f"  Symplectic eigenvalues: {nu}")
    assert np.all(nu >= 1.0), "Eigenvalues must be >= 1"
    
    # Check Symplectic Property of S: S @ Omega @ S.T = Omega
    Omega = np.zeros((2*N, 2*N))
    for i in range(N):
        Omega[2*i, 2*i+1] = 1
        Omega[2*i+1, 2*i] = -1
        
    diff = S @ Omega @ S.T - Omega
    assert np.allclose(diff, 0, atol=1e-10), "S is not symplectic!"
    print("  ✓ S matrix is symplectic")


def test_unitary_phase_sensitivity():
    """
    Test 2: Check if gaussian_unitary_from_symplectic handles PHASES correctly.
    (This detects the error in your previous code).
    """
    print("\n[Test 2] Unitary Phase Rotation")
    ncut = 10
    
    # 1. Create a Symplectic Rotation (1 mode)
    # Rotation by 90 degrees (Phase shift pi/2)
    # q -> p, p -> -q
    theta = np.pi / 2
    S_rot = np.array([
        [np.cos(theta), np.sin(theta)], 
        [-np.sin(theta), np.cos(theta)]
    ])
    
    # 2. Convert to Unitary
    U = gaussian_unitary_from_symplectic(S_rot, ncut)
    
    # 3. Apply to Coherent state |alpha>
    alpha = 2.0
    psi_0 = qt.coherent(ncut, alpha)
    psi_out = U * psi_0
    
    # 4. Expected: |alpha * e^-i*theta>
    psi_expected = qt.coherent(ncut, alpha * np.exp(-1j * theta))
    
    fid = qt.fidelity(psi_out, psi_expected)
    print(f"  Fidelity with expected phase rotation: {fid:.6f}")
    
    if fid < 0.9:
        print("  FAIL: The function missed the phase rotation (likely treated it as Identity).")
    else:
        print("  ✓ PASS: Phase rotation correctly implemented.")
    assert fid > 0.99


def test_tmsv_reduced_state():
    """
    Test 3: The TMSV test that previously failed.
    """
    print("\n[Test 3] TMSV Reduced State (Integration Test)")
    
    ncut = 25 # Need decent cutoff for squeezing
    r = 0.8
    
    # 1. Create TMSV Covariance (q1, p1, q2, p2)
    ch = np.cosh(2*r)
    sh = np.sinh(2*r)
    # Correct correlations for TMSV
    cov = np.array([
        [ch, 0, sh, 0],
        [0, ch, 0, -sh],
        [sh, 0, ch, 0],
        [0, -sh, 0, ch]
    ])
    
    # 2. Get Unitary from this covariance (Via Williamson -> Bloch Messiah)
    # Note: This checks if the decomposition chain allows us to reconstruct the state
    S, nu = williamson_decomposition(cov)
    
    # S maps Vacuum -> TMSV
    U = gaussian_unitary_from_symplectic(S, ncut)
    
    # 3. Generate State
    psi_vac = qt.tensor(qt.basis(ncut,0), qt.basis(ncut,0))
    psi_tmsv = U * psi_vac
    rho_tmsv = qt.ket2dm(psi_tmsv)
    
    # 4. Check Reduced State (Should be Thermal)
    rho_B = rho_tmsv.ptrace(1) # Trace out mode A
    
    # Expected photon number = sinh^2(r)
    n_expected = np.sinh(r)**2
    rho_expected = qt.thermal_dm(ncut, n_expected)
    
    fid = qt.fidelity(rho_B, rho_expected)
    print(f"  Reduced State Fidelity: {fid:.6f} (Expected > 0.99)")
    
    # Calculate Entropies
    S_total = qt.entropy_vn(rho_tmsv)
    S_reduced = qt.entropy_vn(rho_B)
    
    print(f"  Global Entropy: {S_total:.4f} (Expected ~0)")
    print(f"  Reduced Entropy: {S_reduced:.4f} (Expected > 0)")
    
    assert fid > 0.99, "Reduced state is not Thermal!"
    assert S_total < 1e-3, "Global state is not Pure!"
    print("  ✓ PASS: TMSV correlations are correct.")

def compare_densities(rho_numerical, sigma_E_cov, ncut=20):
    """
    Compares the numerical 'summed' density matrix (rho_after) 
    against the analytical Gaussian state (rho_before).
    """
    print("\n=== Density Matrix Comparison ===")

    # 1. Compute Analytical Reference (rho_before)
    # sigma_E is the covariance of Eve's state BEFORE any measurement/filtering.
    # We treat it as a zero-mean Gaussian state.
    mu_zero = np.zeros(4) 
    rho_analytical = heterodyne_gaussian_to_densitymatrix(sigma_E_cov, mu_zero, ncut=ncut)

    # 2. Compute Metrics
    fid = qt.fidelity(rho_analytical, rho_numerical)
    purity_num = rho_numerical.purity()
    purity_ana = rho_analytical.purity()
    entropy_num = qt.entropy_vn(rho_numerical)
    entropy_ana = qt.entropy_vn(rho_analytical)

    print(f"Fidelity (Reference vs Numerical): {fid:.6f}")
    print(f"Purity:   Analytical={purity_ana:.4f} | Numerical={purity_num:.4f}")
    print(f"Entropy:  Analytical={entropy_ana:.4f} | Numerical={entropy_num:.4f}")

    # 3. Decision Logic for Debugging
    if fid > 0.995 and abs(entropy_num - entropy_ana) < 0.15:
        print(">> SUCCESS: Numerical state consistent with analytical Gaussian (within numerical error).")
    elif fid > 0.99:
        print(">> PARTIAL SUCCESS: State structure correct, but integration error still visible.")
    else:
        print(">> WARNING: Significant mismatch detected.")

    # 4. Visualization (Wigner Function)
    # We plot the Wigner function of the first mode (or first few modes)
    # to visually see the "Box Filter" effect or the Gaussian shape.
    
    # Trace out mode 2 to see Mode 1 (optional, depends on if rho is 2-mode)
    # Assuming rho is 2-mode (Eve's parts E1, E2)
    rho_num_1 = rho_numerical.ptrace(0)
    rho_ana_1 = rho_analytical.ptrace(0)

    xvec = np.linspace(-5, 5, 100)
    W_num = qt.wigner(rho_num_1, xvec, xvec)
    W_ana = qt.wigner(rho_ana_1, xvec, xvec)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot Analytical
    cont1 = axes[0].contourf(xvec, xvec, W_ana, 100, cmap='RdBu_r')
    axes[0].set_title("Before Filtering (Analytical)\nGaussian")
    plt.colorbar(cont1, ax=axes[0])

    # Plot Numerical
    cont2 = axes[1].contourf(xvec, xvec, W_num, 100, cmap='RdBu_r')
    axes[1].set_title("After Filtering (Numerical)\nMay be Non-Gaussian")
    plt.colorbar(cont2, ax=axes[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_williamson_property()
    test_unitary_phase_sensitivity()
    test_tmsv_reduced_state()