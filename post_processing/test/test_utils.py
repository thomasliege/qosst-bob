import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import qutip as qt
from utils import (
    williamson_decomposition,
    gaussian_to_densitymatrix,
    gaussian_unitary_from_symplectic
)

N = 2  # number of modes

def test_williamson_decomposition():
    """Test Williamson decomposition on vacuum and thermal states."""
    print("\n=== Testing Williamson Decomposition ===")
    
    # Test 1: Vacuum state
    cov_vacuum = np.eye(2 * N)
    S, nu = williamson_decomposition(cov_vacuum)
    
    print(f"Vacuum covariance:\n{cov_vacuum}")
    print(f"Symplectic eigenvalues: {nu}")
    assert np.allclose(nu, np.ones(N), atol=1e-6), f"Vacuum eigenvalues should be [1,1], got {nu}"
    
    # Verify reconstruction: cov = S.T @ diag([nu,nu]) @ S
    D = np.diag(np.repeat(nu, N))
    cov_recon = S.T @ D @ S
    assert np.allclose(cov_recon, cov_vacuum, atol=1e-6), "Vacuum reconstruction failed"
    print("✓ Vacuum test passed")
    
    # Test 2: Thermal state
    nbar = 2.0  # mean photons per mode
    nu_thermal = np.array([2*nbar + 1, 2*nbar + 1])
    cov_thermal = np.diag(np.repeat(nu_thermal, 2))
    
    S_th, nu_th = williamson_decomposition(cov_thermal)
    print(f"Thermal state (nbar={nbar}) eigenvalues: {nu_th}")
    assert np.allclose(nu_th, nu_thermal, atol=1e-6), f"Thermal eigenvalues mismatch"
    
    cov_th_recon = S_th.T @ np.diag(np.repeat(nu_th, N)) @ S_th
    assert np.allclose(cov_th_recon, cov_thermal, atol=1e-6), "Thermal reconstruction failed"
    print("✓ Thermal state test passed")


def test_gaussian_to_densitymatrix_vacuum():
    """Test Gaussian to density matrix conversion for vacuum."""
    print("\n=== Testing Gaussian to Density Matrix (Vacuum) ===")
    
    ncut = 15
    cov_vacuum = np.eye(2 * N)
    mean_zero = np.zeros(2 * N)
    
    rho = gaussian_to_densitymatrix(cov_vacuum, mean_zero, ncut=ncut)
    
    # Compare with qutip vacuum state
    rho_vac = qt.tensor(qt.basis(ncut, 0) * qt.basis(ncut, 0).dag(),
                        qt.basis(ncut, 0) * qt.basis(ncut, 0).dag())
    
    # Check fidelity
    fidelity = qt.fidelity(rho, rho_vac)
    print(f"Fidelity with |0⟩⊗|0⟩: {fidelity:.6f}")
    assert fidelity > 0.99, f"Low fidelity for vacuum: {fidelity}"
    print("✓ Vacuum density matrix test passed")


def test_gaussian_to_densitymatrix_thermal():
    """Test Gaussian to density matrix conversion for thermal state."""
    print("\n=== Testing Gaussian to Density Matrix (Thermal) ===")
    
    ncut = 20
    nbar = 1.0  # mean photons per mode
    
    # Diagonal covariance for thermal state
    cov_thermal = np.diag([2*nbar + 1, 2*nbar + 1, 2*nbar + 1, 2*nbar + 1])
    mean_zero = np.zeros(2 * N)
    
    rho = gaussian_to_densitymatrix(cov_thermal, mean_zero, ncut=ncut)
    
    # Compare with qutip thermal state
    rho_thermal = qt.tensor(qt.thermal_dm(ncut, nbar), qt.thermal_dm(ncut, nbar))
    
    fidelity = qt.fidelity(rho, rho_thermal)
    print(f"Fidelity with thermal(nbar={nbar}): {fidelity:.6f}")
    assert fidelity > 0.95, f"Low fidelity for thermal: {fidelity}"
    print("✓ Thermal density matrix test passed")


def test_gaussian_to_densitymatrix_displaced():
    """Test Gaussian to density matrix with mean displacement."""
    print("\n=== Testing Gaussian to Density Matrix (Displaced) ===")
    
    ncut = 15
    cov_vacuum = np.eye(2 * N)
    
    # Displace both modes by alpha = 1
    mean = np.array([np.sqrt(2), 0, np.sqrt(2), 0])  # (q1, p1, q2, p2) for alpha=1 in each mode
    
    rho = gaussian_to_densitymatrix(cov_vacuum, mean, ncut=ncut)
    
    # Compare with qutip displaced vacuum
    alpha = 1.0
    rho_disp = qt.tensor(qt.displace(ncut, alpha) * qt.basis(ncut, 0),
                         qt.displace(ncut, alpha) * qt.basis(ncut, 0))
    rho_disp = rho_disp * rho_disp.dag()
    
    fidelity = qt.fidelity(rho, rho_disp)
    print(f"Fidelity with displaced vacuum: {fidelity:.6f}")
    assert fidelity > 0.90, f"Low fidelity for displaced state: {fidelity}"
    print("✓ Displaced density matrix test passed")


def test_gaussian_unitary_from_symplectic():
    """Test Bloch-Messiah unitary construction."""
    print("\n=== Testing Gaussian Unitary from Symplectic ===")
    
    ncut = 15
    
    # Test 1: Identity (no squeezing/rotation)
    S_identity = np.eye(2 * N)
    U_id = gaussian_unitary_from_symplectic(S_identity, ncut=ncut)
    
    # Apply to vacuum and check it's still vacuum
    rho_vac = qt.tensor(qt.basis(ncut, 0) * qt.basis(ncut, 0).dag(),
                        qt.basis(ncut, 0) * qt.basis(ncut, 0).dag())
    rho_transformed = U_id * rho_vac * U_id.dag()
    
    fidelity = qt.fidelity(rho_transformed, rho_vac)
    print(f"Identity unitary fidelity: {fidelity:.6f}")
    assert fidelity > 0.99, f"Identity unitary failed: {fidelity}"
    print("✓ Identity unitary test passed")
    
    # Test 2: Squeezed state (use Williamson decomposition)
    nbar = 0.5
    r = 0.5  # squeezing parameter
    # Create a squeezed thermal covariance
    cov_squeezed = np.diag([np.exp(-2*r) * (2*nbar + 1),
                            np.exp(2*r) * (2*nbar + 1),
                            np.exp(-2*r) * (2*nbar + 1),
                            np.exp(2*r) * (2*nbar + 1)])
    
    # Get symplectic matrix via Williamson
    S_squeezed, nu_squeezed = williamson_decomposition(cov_squeezed)
    U_squeeze = gaussian_unitary_from_symplectic(S_squeezed, ncut=ncut)
    
    # Check it's unitary
    U_dag_U = U_squeeze.dag() * U_squeeze
    assert np.allclose(U_dag_U.full(), np.eye(ncut**2), atol=1e-6), "Unitary condition violated"
    print("✓ Squeezing unitary is valid")

def test_consistency():
    """Test consistency between different conversion methods."""
    print("\n=== Testing Consistency ===")
    
    ncut = 15
    nbar = 0.5
    cov_thermal = np.diag([2*nbar + 1] * 4)
    mean = np.zeros(4)
    
    # Method 1: Direct gaussian_to_densitymatrix
    rho1 = gaussian_to_densitymatrix(cov_thermal, mean, ncut=ncut)
    
    # Method 2: Manual construction via Williamson + unitary
    S, nu = williamson_decomposition(cov_thermal)
    nbar_w = np.maximum((nu - 1.0) / 2.0, 0.0)
    rho_th = qt.tensor(qt.thermal_dm(ncut, nbar_w[0]), qt.thermal_dm(ncut, nbar_w[1]))
    U = gaussian_unitary_from_symplectic(S, ncut=ncut)
    rho2 = U * rho_th * U.dag()
    
    # Compare
    fidelity = qt.fidelity(rho1, rho2)
    print(f"Fidelity between methods: {fidelity:.6f}")
    assert fidelity > 0.98, f"Methods inconsistent: {fidelity}"
    print("✓ Consistency test passed")


if __name__ == "__main__":
    test_williamson_decomposition()
    test_gaussian_to_densitymatrix_vacuum()
    test_gaussian_to_densitymatrix_thermal()
    test_gaussian_to_densitymatrix_displaced()
    test_gaussian_unitary_from_symplectic()
    test_consistency()
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
