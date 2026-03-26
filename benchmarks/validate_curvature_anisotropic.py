#!/usr/bin/env python3
"""E1: Validate all 14 Florinsky curvatures on an anisotropic sinusoidal ridge surface.

Surface: z(x,y) = A * sin(kx*x) * cos(ky*y)

This surface produces non-zero values for ALL 14 curvatures, including
k_ve, K_r, and rotor which are identically zero on radially symmetric surfaces.

Analytical derivatives are computed from closed-form partial derivatives.
"""

import numpy as np
import sys

def create_sinusoidal_ridge(rows, cols, cell_size, A, lambda_x, lambda_y):
    """Create sinusoidal ridge surface z = A * sin(kx*x) * cos(ky*y).

    Uses kx != ky to ensure anisotropy (all 14 curvatures non-trivial).
    """
    kx = 2 * np.pi / lambda_x
    ky = 2 * np.pi / lambda_y

    # Center the grid
    y_coords = (np.arange(rows) - rows / 2) * cell_size
    x_coords = (np.arange(cols) - cols / 2) * cell_size
    X, Y = np.meshgrid(x_coords, y_coords)

    Z = A * np.sin(kx * X) * np.cos(ky * Y)

    return X, Y, Z, kx, ky


def analytical_curvatures(X, Y, A, kx, ky):
    """Compute all 14 Florinsky curvatures analytically.

    Returns dict of curvature_name -> 2D array.
    """
    S = np.sin(kx * X)
    C = np.cos(kx * X)
    Sy = np.sin(ky * Y)
    Cy = np.cos(ky * Y)

    # Partial derivatives in GEOGRAPHIC convention (y upward = -Y_row)
    # SurtGIS uses q = (z_above - z_below) / cs6, which is dz/dy_geographic
    # Since Y in this grid increases downward (with row), we need:
    #   q_geo = -dz/dY_row = A ky S Sy  (sign flipped)
    #   s_geo = -d²z/dXdY_row = A kx ky C Sy  (sign flipped)
    # p, r, t are unaffected (x convention is the same)
    p = A * kx * C * Cy           # dz/dx (same in both)
    q = A * ky * S * Sy            # dz/dy_geographic = -dz/dY_row
    r = -A * kx**2 * S * Cy       # d²z/dx² (same)
    s = A * kx * ky * C * Sy       # d²z/dxdy_geographic
    t = -A * ky**2 * S * Cy       # d²z/dy² (same, second deriv sign cancels)

    # Intermediate quantities
    p2 = p**2
    q2 = q**2
    G = p2 + q2  # gradient squared
    w = 1.0 + G  # 1 + p² + q²

    # Avoid division by zero at flat points
    eps = 1e-30
    G_safe = np.where(G > eps, G, eps)

    # 1. Mean curvature H
    num_H = -((1 + q2) * r - 2 * p * q * s + (1 + p2) * t)
    H = num_H / (2 * w**1.5)

    # 2. Gaussian curvature K
    K = (r * t - s**2) / w**2

    # 3. Horizontal curvature kh
    num_kh = -(q2 * r - 2 * p * q * s + p2 * t)
    kh = num_kh / (G_safe * np.sqrt(w))

    # 4. Vertical curvature kv
    num_kv = -(p2 * r + 2 * p * q * s + q2 * t)
    kv = num_kv / (G_safe * w**1.5)

    # 5. Laplacian
    laplacian = r + t

    # 6. Unsphericity M
    H2_minus_K = H**2 - K
    H2_minus_K = np.maximum(H2_minus_K, 0.0)  # numerical guard
    M = np.sqrt(H2_minus_K)

    # 7. Difference E
    E = (kv - kh) / 2.0

    # 8-9. Principal curvatures
    kmin = H - M
    kmax = H + M

    # 10-11. Excess curvatures
    khe = M - E
    kve = M + E

    # 12. Accumulation curvature
    Ka = kh * kv

    # 13. Ring curvature
    Kr = M**2 - E**2

    # 14. Rotor
    num_rot = (p2 - q2) * s - p * q * (r - t)
    rot = num_rot / G_safe**1.5

    # Set flat points (G ~ 0) to NaN
    flat_mask = G < eps
    for arr in [kh, kv, rot]:
        arr[flat_mask] = np.nan

    return {
        'mean_h': H,
        'gaussian_k': K,
        'unsphericity_m': M,
        'laplacian': laplacian,
        'minimal_kmin': kmin,
        'maximal_kmax': kmax,
        'difference_e': E,
        'accumulation_ka': Ka,
        'horizontal_kh': kh,
        'vertical_kv': kv,
        'horizontal_excess_khe': khe,
        'vertical_excess_kve': kve,
        'ring_kr': Kr,
        'rotor': rot,
    }


def run_surtgis_curvatures(dem, cell_size):
    """Run SurtGIS advanced_curvature for all 14 types."""
    import surtgis

    ctypes = [
        'mean_h', 'gaussian_k', 'unsphericity_m', 'laplacian',
        'minimal_kmin', 'maximal_kmax', 'difference_e', 'accumulation_ka',
        'horizontal_kh', 'vertical_kv', 'horizontal_excess_khe',
        'vertical_excess_kve', 'ring_kr', 'rotor',
    ]

    results = {}
    for ctype in ctypes:
        results[ctype] = surtgis.advanced_curvature(dem, cell_size, ctype=ctype)

    return results


def compare_curvatures(analytical, surtgis_results, X, Y, A, kx, ky, border=2):
    """Compare analytical vs SurtGIS curvatures, excluding border cells.

    For gradient-dependent curvatures (kh, kv, rotor, etc.), additionally
    masks out cells where |gradient|² < threshold to avoid singularities.
    """
    # Compute gradient magnitude for masking
    S = np.sin(kx * X)
    C = np.cos(kx * X)
    Sy = np.sin(ky * Y)
    Cy = np.cos(ky * Y)
    p = A * kx * C * Cy
    q = -A * ky * S * Sy
    G = p**2 + q**2
    G_inner = G[border:-border, border:-border]

    # Gradient-dependent curvatures that have singularities at G→0
    gradient_dep = {'horizontal_kh', 'vertical_kv', 'rotor',
                    'difference_e', 'horizontal_excess_khe',
                    'vertical_excess_kve', 'accumulation_ka', 'ring_kr'}

    # Threshold: exclude bottom 10% of gradient values
    g_threshold = np.percentile(G_inner[G_inner > 0], 10)

    print(f"\n{'Curvature':<25} {'RMSE':<18} {'MAE':<18} {'R²':<12} {'N valid':<10} {'Range (analytical)'}")
    print("-" * 110)

    results = []

    for name in analytical:
        ana = analytical[name]
        sg = surtgis_results[name]

        # Exclude border cells and NaN/Inf
        inner_ana = ana[border:-border, border:-border]
        inner_sg = sg[border:-border, border:-border]

        mask = np.isfinite(inner_ana) & np.isfinite(inner_sg)

        # For gradient-dependent curvatures, also exclude low-gradient cells
        if name in gradient_dep:
            mask &= (G_inner > g_threshold)

        n_valid = np.sum(mask)

        if n_valid == 0:
            print(f"{name:<25} {'N/A':<18} {'N/A':<18} {'N/A':<12}")
            continue

        a = inner_ana[mask]
        s = inner_sg[mask]

        diff = a - s
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))

        ss_res = np.sum(diff**2)
        ss_tot = np.sum((a - np.mean(a))**2)

        if ss_tot > 1e-30:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = float('nan')

        ana_range = np.max(a) - np.min(a)
        ana_min = np.min(a)
        ana_max = np.max(a)

        # Check if signal is non-trivial
        is_nontrivial = ana_range > 1e-10

        r2_str = f"{r2:.6f}" if not np.isnan(r2) else "---"
        grad_note = " *" if name in gradient_dep else ""

        print(f"{name:<25} {rmse:<18.6e} {mae:<18.6e} {r2_str:<12} {n_valid:<10,} [{ana_min:.4e}, {ana_max:.4e}]{grad_note}")

        results.append({
            'name': name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_valid': n_valid,
            'ana_range': ana_range,
            'is_nontrivial': is_nontrivial,
        })

    print("\n* Gradient-dependent: low-gradient cells (bottom 5%) excluded to avoid singularities")

    return results


def main():
    print("=" * 60)
    print("E1: Curvature validation on anisotropic sinusoidal ridge")
    print("=" * 60)

    # Surface parameters
    # Use larger grid with finer cell size for better finite-difference accuracy
    rows, cols = 2000, 2000
    cell_size = 5.0  # meters (finer = better finite-difference accuracy)
    A = 500.0  # amplitude (meters)
    lambda_x = 4000.0  # wavelength in x (meters) — ~800 cells
    lambda_y = 3000.0  # wavelength in y (meters) — ~600 cells (anisotropic!)

    kx = 2 * np.pi / lambda_x
    ky = 2 * np.pi / lambda_y

    print(f"\nSurface: z = {A} * sin({kx:.4f} x) * cos({ky:.4f} y)")
    print(f"Grid: {rows}x{cols}, cell_size = {cell_size} m")
    print(f"Wavelengths: λx = {lambda_x} m ({lambda_x/cell_size:.0f} cells), "
          f"λy = {lambda_y} m ({lambda_y/cell_size:.0f} cells)")
    print(f"kx/ky ratio: {kx/ky:.3f} (≠ 1 → anisotropic)")

    # Create surface
    print("\nCreating sinusoidal ridge surface...")
    X, Y, Z, kx, ky = create_sinusoidal_ridge(rows, cols, cell_size, A, lambda_x, lambda_y)
    print(f"Z range: [{Z.min():.1f}, {Z.max():.1f}] m")

    # Analytical curvatures
    print("Computing analytical curvatures...")
    ana = analytical_curvatures(X, Y, A, kx, ky)

    # Check that previously-zero quantities are now non-trivial
    print("\nNon-triviality check (were zero on Gaussian hill):")
    for name in ['vertical_excess_kve', 'ring_kr', 'rotor']:
        arr = ana[name]
        valid = arr[np.isfinite(arr)]
        rng = np.max(valid) - np.min(valid) if len(valid) > 0 else 0
        print(f"  {name}: range = {rng:.4e}, "
              f"mean abs = {np.mean(np.abs(valid)):.4e} "
              f"{'✓ NON-TRIVIAL' if rng > 1e-6 else '✗ NEAR-ZERO'}")

    # SurtGIS curvatures
    print("\nComputing SurtGIS curvatures (14 types)...")
    sg = run_surtgis_curvatures(Z, cell_size)

    # Compare
    results = compare_curvatures(ana, sg, X, Y, A, kx, ky)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_nontrivial = sum(1 for r in results if r['is_nontrivial'])
    n_good_r2 = sum(1 for r in results if r['is_nontrivial'] and not np.isnan(r['r2']) and r['r2'] > 0.999)
    max_rmse = max(r['rmse'] for r in results)

    print(f"Non-trivial curvatures: {n_nontrivial}/14")
    print(f"R² > 0.999 (non-trivial): {n_good_r2}/{n_nontrivial}")
    print(f"Maximum RMSE: {max_rmse:.6e}")

    # Check for the 3 previously-zero types
    prev_zero = ['vertical_excess_kve', 'ring_kr', 'rotor']
    print(f"\nPreviously-zero curvatures (now validated):")
    for name in prev_zero:
        r = next(r for r in results if r['name'] == name)
        status = "✓" if r['is_nontrivial'] and (np.isnan(r['r2']) or r['r2'] > 0.99) else "✗"
        r2_str = f"R²={r['r2']:.6f}" if not np.isnan(r['r2']) else "R²=---"
        print(f"  {status} {name}: RMSE={r['rmse']:.4e}, {r2_str}, range={r['ana_range']:.4e}")


if __name__ == "__main__":
    main()
