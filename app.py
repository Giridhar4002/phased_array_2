"""
================================================================================
CICAD 2025 — Phased Array Antenna Design & Analysis Tool
================================================================================
Problem 2: Hexagonal-grid phased array at 44.5 GHz ± 1 GHz
  • Potter horn elements, circular aperture, 10 dB amplitude taper
  • ±9° scan, 40 dBi minimum gain over coverage
  • Part D: complexity reduction at ±6° scan

================================================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import math

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
C_LIGHT = 0.299792458  # speed of light in m/ns  →  gives λ in metres when f in GHz
PI = np.pi

# Radiating-element beamwidth constant A (degrees) — Eq. (7)
ELEMENT_BW_CONSTANTS = {
    "High-efficiency Multimode Horn": 63,
    "Potter Horn": 70,
    "Corrugated Horn": 75,
    "Cup-dipole Radiating Element": 58,
    "Dominant-mode Square Horn": 55,
    "High-efficiency Square/Rectangular Horn": 52,
    "Patch Antenna": 58,
    "Dipole": 58,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def wavelength_m(freq_ghz: float) -> float:
    """Return free-space wavelength in metres for a frequency in GHz."""
    return C_LIGHT / freq_ghz


def taper_efficiency(T_dB: float) -> float:
    """
    Aperture-illumination efficiency η for parabolic-on-pedestal taper.
    Eq. (10):  η = 75 · (1+T)² / (1+T+T²)   [%]
    where T = 10^(−taper_dB / 20)  is the linear pedestal voltage.
    Returns fractional efficiency (0–1).
    """
    T_lin = 10 ** (-abs(T_dB) / 20.0)
    eta_pct = 75.0 * (1 + T_lin) ** 2 / (1 + T_lin + T_lin ** 2)
    return eta_pct / 100.0


def hex_element_spacing(lambda_h: float, theta_sm_deg: float,
                        theta_G_deg: float) -> float:
    """
    Max element spacing for hexagonal lattice — Eq. (2):
        d_h / λ_h  ≤  1.1547 / (sin θ_sm + sin θ_G)
    Returns spacing in metres.
    """
    sin_sum = math.sin(math.radians(theta_sm_deg)) + math.sin(math.radians(theta_G_deg))
    d_over_lambda = 1.1547 / sin_sum
    return d_over_lambda * lambda_h, d_over_lambda


def square_element_spacing(lambda_h: float, theta_sm_deg: float,
                           theta_G_deg: float) -> float:
    """
    Max element spacing for square lattice — Eq. (1):
        d_s / λ_h  ≤  1 / (sin θ_sm + sin θ_G)
    """
    sin_sum = math.sin(math.radians(theta_sm_deg)) + math.sin(math.radians(theta_G_deg))
    d_over_lambda = 1.0 / sin_sum
    return d_over_lambda * lambda_h, d_over_lambda


def element_directivity_dBi(eta_e: float, spacing_m: float,
                            lambda_l: float) -> float:
    """
    Element directivity — Eq. (3) component:
        D_e = η_e · 4π · A_e / λ_l²
    For hexagonal lattice the unit-cell area is A_e = (√3/2)·d².
    lambda_l is the wavelength at the *lowest* frequency (governs directivity).
    """
    A_e = (math.sqrt(3) / 2.0) * spacing_m ** 2
    D_lin = eta_e * 4.0 * PI * A_e / lambda_l ** 2
    return 10.0 * math.log10(D_lin), D_lin, A_e


def scan_loss_dB(theta_sm_deg: float, spacing_norm: float,
                 element_bw_const: float) -> float:
    """
    Scan loss — Eq. (6) for directive elements (d/λ > 1):
        SL = 3 · (θ_sm / (0.5·θ_3))²
    where θ_3 = A · (λ_h / d_e) = A / (d/λ).
    For d/λ < 1 use cos^n model Eq. (8) with n = 1.5.
    """
    if spacing_norm >= 1.0:
        theta_3 = element_bw_const / spacing_norm  # degrees
        sl = 3.0 * (theta_sm_deg / (0.5 * theta_3)) ** 2
    else:
        n = 1.5
        sl = -10.0 * math.log10(math.cos(math.radians(theta_sm_deg)) ** n)
    return sl


def grating_lobe_angle(theta_sm_deg: float) -> float:
    """
    Select grating-lobe placement just outside scan region.
    Design rules used:
        θ_sm ≤ 15°  → θ_G = θ_sm + 1
        θ_sm ≤ 45°  → θ_G = θ_sm + 2
        θ_sm > 60°  → θ_G = θ_sm + 5
    """
    if theta_sm_deg <= 15:
        return theta_sm_deg + 1.0
    elif theta_sm_deg <= 45:
        return theta_sm_deg + 2.0
    else:
        return theta_sm_deg + 5.0


def peak_directivity_dBi(G_min_dBi: float, SL_dB: float, TL_dB: float,
                         L_s_dB: float, GL_pe_dB: float, X_dB: float,
                         I_m_dB: float) -> float:
    """
    Required peak directivity — Eq. (5):
        D_p = G_min + L_s + SL + GL_pe + T_L + X + I_m
    All values in dB.
    """
    return G_min_dBi + L_s_dB + SL_dB + GL_pe_dB + TL_dB + X_dB + I_m_dB


def num_elements(Dp_dBi: float, De_dBi: float) -> float:
    """
    Number of elements — Eq. (4):
        N = 10^(0.1·D_p − 0.1·D_e)
    """
    return 10 ** (0.1 * Dp_dBi - 0.1 * De_dBi)


def array_directivity_dBi(N: int, eta_taper: float, De_lin: float) -> float:
    """
    Recalculate exact directivity from integer N — Eq. (3):
        D_p = 10·log10[η_taper · N · D_e_lin]
    """
    return 10.0 * math.log10(eta_taper * N * De_lin)


def generate_hex_grid_circular(d_m: float, R_m: float):
    """
    Generate element positions on a hexagonal grid inside a circle of radius R_m.
    Returns arrays (x, y) in metres.
    """
    # Row spacing for hex grid
    dy = d_m * math.sqrt(3) / 2.0
    dx = d_m

    # Determine grid bounds
    n_rows = int(math.ceil(R_m / dy)) + 1
    positions = []
    for row in range(-n_rows, n_rows + 1):
        y = row * dy
        # Offset every other row
        x_offset = 0.5 * dx if (row % 2 != 0) else 0.0
        n_cols = int(math.ceil(R_m / dx)) + 1
        for col in range(-n_cols, n_cols + 1):
            x = col * dx + x_offset
            if x ** 2 + y ** 2 <= R_m ** 2:
                positions.append((x, y))
    positions = np.array(positions)
    return positions[:, 0], positions[:, 1]


def compute_array_factor(theta_deg, phi_deg, x_pos, y_pos, weights,
                         lambda_0):
    """
    Compute array factor AF(θ) for given element positions and weights.
    θ is elevation from boresight, φ is azimuth cut (fixed).
    """
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    kx = (2 * PI / lambda_0) * np.sin(theta_rad) * np.cos(phi_rad)
    ky = (2 * PI / lambda_0) * np.sin(theta_rad) * np.sin(phi_rad)

    AF = np.zeros_like(theta_rad, dtype=complex)
    for i in range(len(x_pos)):
        phase = kx * x_pos[i] + ky * y_pos[i]
        AF += weights[i] * np.exp(1j * phase)
    return AF


def gaussian_taper_weights(x, y, R, taper_dB):
    """
    Apply a Gaussian (parabolic-on-pedestal approximation) taper.
    Elements at the edge are taper_dB below the centre.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    r_norm = r / R  # 0 at centre, 1 at edge
    # Pedestal voltage
    T_lin = 10 ** (-abs(taper_dB) / 20.0)
    # Parabolic on pedestal: E(r) = T + (1-T)(1 - r²)  for n=1
    weights = T_lin + (1.0 - T_lin) * (1.0 - r_norm ** 2)
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Full design procedure
# ──────────────────────────────────────────────────────────────────────────────

def run_design(f_center, f_offset, theta_sm, G_min, taper_dB, eta_e_pct,
               element_name, L_s, GL_pe, X, I_m):
    """
    Execute the full phased-array design and return a results dict.
    """
    res = {}

    # Derived frequencies
    f_max = f_center + f_offset   # GHz — worst-case for grating lobes
    f_min = f_center - f_offset   # GHz — worst-case for directivity
    lambda_h = wavelength_m(f_max)          # shortest λ (grating-lobe calc)
    lambda_l = wavelength_m(f_min)          # longest  λ (directivity calc)
    lambda_c = wavelength_m(f_center)       # centre   λ (nominal)

    res["f_center"] = f_center
    res["f_min"] = f_min
    res["f_max"] = f_max
    res["lambda_h_mm"] = lambda_h * 1000
    res["lambda_l_mm"] = lambda_l * 1000
    res["lambda_c_mm"] = lambda_c * 1000

    # Element beamwidth constant
    A_const = ELEMENT_BW_CONSTANTS.get(element_name, 70)

    # Taper efficiency
    eta_taper = taper_efficiency(taper_dB)
    res["eta_taper"] = eta_taper
    # Taper loss in dB
    TL_dB = -10.0 * math.log10(eta_taper)
    res["TL_dB"] = round(TL_dB, 3)

    # Element efficiency (fractional)
    eta_e = eta_e_pct / 100.0

    # Grating-lobe placement
    theta_G = grating_lobe_angle(theta_sm)
    res["theta_G"] = theta_G

    # Element spacing (hexagonal lattice)
    d_hex_m, d_hex_norm = hex_element_spacing(lambda_h, theta_sm, theta_G)
    res["d_hex_m"] = d_hex_m
    res["d_hex_mm"] = d_hex_m * 1000
    res["d_hex_norm"] = d_hex_norm  # d / λ_h

    # Element directivity (use λ_l for directivity)
    De_dBi, De_lin, A_e = element_directivity_dBi(eta_e, d_hex_m, lambda_l)
    res["De_dBi"] = De_dBi
    res["De_lin"] = De_lin
    res["A_e_mm2"] = A_e * 1e6

    # Scan loss
    SL = scan_loss_dB(theta_sm, d_hex_norm, A_const)
    res["SL_dB"] = SL

    # Required peak directivity  — Eq. (5)
    Dp = peak_directivity_dBi(G_min, SL, TL_dB, L_s, GL_pe, X, I_m)
    res["Dp_dBi"] = Dp

    # Number of elements
    N_raw = num_elements(Dp, De_dBi)
    N_int = math.ceil(N_raw)
    res["N_raw"] = N_raw
    res["N"] = N_int

    # Recalculated directivity with integer N
    Dp_actual = array_directivity_dBi(N_int, eta_taper, De_lin)
    res["Dp_actual_dBi"] = Dp_actual
    Dp_at_scan = Dp_actual - SL
    res["Dp_at_scan_dBi"] = Dp_at_scan

    # Gain at scan edge (subtract losses)
    G_at_scan = Dp_at_scan - L_s - GL_pe - I_m
    res["G_at_scan_dBi"] = G_at_scan

    # Array physical size
    total_area = N_int * A_e
    R_circ = math.sqrt(total_area / PI)
    diameter_circ = 2 * R_circ
    res["total_area_m2"] = total_area
    res["R_circ_m"] = R_circ
    res["diameter_circ_m"] = diameter_circ
    res["diameter_circ_lambda"] = diameter_circ / lambda_c

    # Generate element positions
    x_pos, y_pos = generate_hex_grid_circular(d_hex_m, R_circ)
    N_actual = len(x_pos)
    res["N_placed"] = N_actual

    # If placed elements differ significantly, re-adjust
    if N_actual > 0:
        Dp_placed = array_directivity_dBi(N_actual, eta_taper, De_lin)
    else:
        Dp_placed = 0
    res["Dp_placed_dBi"] = Dp_placed

    # Taper weights
    weights = gaussian_taper_weights(x_pos, y_pos, R_circ, taper_dB)

    # Grating-lobe locations
    # Boresight GL
    if 1.1547 / d_hex_norm < 1.0:
        boresight_GL = 90.0
    else:
        val = 1.1547 / d_hex_norm
        if val <= 1.0:
            boresight_GL = np.degrees(np.arcsin(val))
        else:
            boresight_GL = 90.0
    # Scanned GL
    val2 = 1.1547 / d_hex_norm - np.sin(np.radians(theta_sm))
    if abs(val2) <= 1.0:
        scan_GL = np.degrees(np.arcsin(val2))
    else:
        scan_GL = 90.0
    res["boresight_GL_deg"] = boresight_GL
    res["scan_GL_deg"] = scan_GL

    # Half-power beamwidth of element
    theta_3_element = A_const / d_hex_norm  # degrees
    res["theta_3_element"] = theta_3_element

    # Pack arrays for plotting
    res["x_pos"] = x_pos
    res["y_pos"] = y_pos
    res["weights"] = weights
    res["lambda_c"] = lambda_c
    res["R_circ"] = R_circ
    res["eta_e"] = eta_e
    res["taper_dB"] = taper_dB

    return res


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_array_layout(x, y, weights, R, d_m, title="Array Layout"):
    """Scatter plot of element positions coloured by taper weight."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Aperture circle
    circle = plt.Circle((0, 0), R * 1000, fill=False, edgecolor="#4fc3f7",
                         linewidth=1.5, linestyle="--", label="Aperture boundary")
    ax.add_patch(circle)

    sc = ax.scatter(x * 1000, y * 1000, c=weights, cmap="inferno",
                    s=18, edgecolors="none", zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label("Taper Weight (linear)", color="#111111", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#111111")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#111111")

    ax.set_xlabel("x  (mm)", color="#111111", fontsize=11)
    ax.set_ylabel("y  (mm)", color="#111111", fontsize=11)
    ax.set_title(title, color="#111111", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(loc="upper right", fontsize=9, facecolor="#ffffff",
              edgecolor="#bbbbbb", labelcolor="#111111")
    ax.grid(True, alpha=0.15, color="#bbbbbb")
    fig.tight_layout()
    return fig


def plot_radiation_pattern(x_pos, y_pos, weights, lambda_0, Dp_dBi,
                           title="Array Radiation Pattern"):
    """Compute and plot the normalized array factor in a principal plane."""
    theta_range = np.linspace(-30, 30, 3001)
    AF = compute_array_factor(theta_range, 0.0, x_pos, y_pos, weights, lambda_0)
    AF_mag = np.abs(AF)
    AF_mag_max = AF_mag.max()
    if AF_mag_max > 0:
        AF_norm_dB = 20 * np.log10(AF_mag / AF_mag_max)
    else:
        AF_norm_dB = np.zeros_like(AF_mag)

    # Scale to absolute directivity
    AF_abs_dB = AF_norm_dB + Dp_dBi

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.plot(theta_range, AF_abs_dB, color="#00e5ff", linewidth=1.2,
            label="φ = 0° cut")
    ax.axhline(y=Dp_dBi, color="#ff9800", linestyle=":", linewidth=0.8,
               label=f"Peak = {Dp_dBi:.1f} dBi")
    ax.axhline(y=Dp_dBi - 3, color="#66bb6a", linestyle="--", linewidth=0.8,
               label="−3 dB")
    ax.set_ylim(Dp_dBi - 50, Dp_dBi + 3)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("θ  (degrees)", color="#111111", fontsize=11)
    ax.set_ylabel("Directivity  (dBi)", color="#111111", fontsize=11)
    ax.set_title(title, color="#111111", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(fontsize=9, facecolor="#ffffff", edgecolor="#bbbbbb",
              labelcolor="#111111")
    ax.grid(True, alpha=0.15, color="#bbbbbb")
    fig.tight_layout()
    return fig


def plot_normalized_pattern(x_pos, y_pos, weights, lambda_0, Dp_dBi,
                            title="Normalized Radiation Pattern"):
    """Plot the normalized (0 dB peak) pattern."""
    theta_range = np.linspace(-30, 30, 3001)
    AF = compute_array_factor(theta_range, 0.0, x_pos, y_pos, weights, lambda_0)
    AF_mag = np.abs(AF)
    AF_mag_max = AF_mag.max()
    if AF_mag_max > 0:
        AF_norm_dB = 20 * np.log10(AF_mag / AF_mag_max)
    else:
        AF_norm_dB = np.zeros_like(AF_mag)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.plot(theta_range, AF_norm_dB, color="#ce93d8", linewidth=1.2)
    ax.axhline(y=-3, color="#66bb6a", linestyle="--", linewidth=0.8, label="−3 dB")
    ax.set_ylim(-50, 3)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("θ  (degrees)", color="#111111", fontsize=11)
    ax.set_ylabel("Normalized Gain  (dB)", color="#111111", fontsize=11)
    ax.set_title(title, color="#111111", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(fontsize=9, facecolor="#ffffff", edgecolor="#bbbbbb",
              labelcolor="#111111")
    ax.grid(True, alpha=0.15, color="#bbbbbb")
    fig.tight_layout()
    return fig


def plot_efficiency_vs_taper():
    """Plot array efficiency vs illumination taper."""
    taper_range = np.linspace(0, 20, 200)
    eff = np.array([taper_efficiency(t) * 100 for t in taper_range])

    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.plot(taper_range, eff, color="#ffd54f", linewidth=1.5)
    ax.axvline(x=10, color="#ef5350", linestyle="--", linewidth=0.8,
               label="10 dB taper")
    ax.set_xlabel("Edge Illumination Taper  (dB)", color="#111111", fontsize=11)
    ax.set_ylabel("Array Efficiency  (%)", color="#111111", fontsize=11)
    ax.set_title("Array Efficiency vs Illumination Taper", color="#111111",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(fontsize=9, facecolor="#ffffff", edgecolor="#bbbbbb",
              labelcolor="#111111")
    ax.grid(True, alpha=0.15, color="#bbbbbb")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CICAD 2025 — PA Antenna Design",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* light metric cards with dark text */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetric"] * {
        color: #111111 !important;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-signal.png", width=64)
    st.title("🛰️ Input Parameters")
    st.markdown("---")

    f_center = st.number_input("Centre Frequency (GHz)", min_value=1.0,
                               value=44.5, step=0.5, format="%.2f")
    f_offset = st.number_input("Frequency Offset ± (GHz)", min_value=0.0,
                               value=1.0, step=0.1, format="%.2f",
                               help="Bandwidth = ±offset → max freq used for grating-lobe calc")

    st.markdown("---")
    theta_sm = st.number_input("Max Scan Angle (degrees)", min_value=0.0,
                               max_value=90.0, value=9.0, step=1.0)
    G_min = st.number_input("Min Gain over Coverage (dBi)", min_value=0.0,
                            value=40.0, step=1.0)
    taper_dB = st.number_input("Edge Illumination Taper (dB)", min_value=0.0,
                               value=10.0, step=1.0)
    eta_e_pct = st.number_input("Element Aperture Efficiency (%)",
                                min_value=1.0, max_value=100.0, value=70.0,
                                step=5.0)
    element_name = st.selectbox("Radiating Element Type",
                                list(ELEMENT_BW_CONSTANTS.keys()), index=1)

    st.markdown("---")
    st.markdown("**Additional Losses (dB)**")
    L_s = st.number_input("Antenna / Front-end Loss", min_value=0.0,
                          value=0.0, step=0.1)
    GL_pe = st.number_input("Pointing Error Loss", min_value=0.0,
                            value=0.0, step=0.1)
    X = st.number_input("Loss over Beam Diameter", min_value=0.0,
                        value=0.0, step=0.1,
                        help="Typically 3 dB if full beam, 0 dB for peak only")
    I_m = st.number_input("Implementation Margin", min_value=0.0,
                          value=0.5, step=0.1)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📡  CICAD 2025 — Phased Array Antenna Design Tool")
st.markdown(
    f"**Problem 2 :** Hexagonal-grid PA at **{f_center} GHz ± {f_offset} GHz**, "
    f"**{element_name}** elements, circular aperture, "
    f"**{taper_dB:.0f} dB** taper, **±{theta_sm:.0f}°** scan, "
    f"**{G_min:.0f} dBi** min gain."
)
st.markdown("---")

# ── Run design ────────────────────────────────────────────────────────────────
res = run_design(f_center, f_offset, theta_sm, G_min, taper_dB, eta_e_pct,
                 element_name, L_s, GL_pe, X, I_m)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION A — Peak Directivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Part A — Required Peak Directivity")

st.markdown(
    "Using **Eq. (5)**:\n\n"
    r"$$D_p = G_{\min} + L_s + SL + GL_{pe} + T_L + X + I_m$$"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("λ_min (mm)", f"{res['lambda_h_mm']:.3f}")
c2.metric("λ_max (mm)", f"{res['lambda_l_mm']:.3f}")
c3.metric("λ_center (mm)", f"{res['lambda_c_mm']:.3f}")
c4.metric("Grating Lobe θ_G", f"{res['theta_G']:.1f}°")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Taper Efficiency η_taper", f"{res['eta_taper']*100:.2f} %")
c2.metric("Taper Loss T_L", f"{res['TL_dB']:.2f} dB")
c3.metric("Scan Loss SL", f"{res['SL_dB']:.2f} dB")

st.markdown("---")

st.markdown("### ✅  Required Peak Directivity")
c1, c2 = st.columns(2)
c1.metric("D_p (required)", f"{res['Dp_dBi']:.2f} dBi")
c2.metric("Breakdown",
          f"{G_min:.1f} + {L_s:.1f} + {res['SL_dB']:.2f} + "
          f"{GL_pe:.1f} + {res['TL_dB']:.2f} + {X:.1f} + {I_m:.1f} dB")

st.info(
    f"To achieve **{G_min:.0f} dBi** minimum gain at the ±{theta_sm:.0f}° scan edge "
    f"with a {taper_dB:.0f} dB taper and {eta_e_pct:.0f}% element efficiency, "
    f"the array peak boresight directivity must be at least **{res['Dp_dBi']:.2f} dBi**."
)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION B — Spacing & Elements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Part B — Element Spacing, Element Gain & Number of Elements")

st.markdown(
    "**Hexagonal lattice spacing** — Eq. (2):\n\n"
    r"$$\frac{d_h}{\lambda_h} \leq \frac{1.1547}{\sin\theta_{sm} + \sin\theta_G}$$"
)

c1, c2, c3 = st.columns(3)
c1.metric("d / λ_h", f"{res['d_hex_norm']:.4f}")
c2.metric("d (mm)", f"{res['d_hex_mm']:.3f}")
c3.metric("Unit-cell area (mm²)", f"{res['A_e_mm2']:.3f}")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Element Directivity D_e", f"{res['De_dBi']:.2f} dBi")
c2.metric("Element Gain G_e", f"{res['De_dBi'] - 10*math.log10(1/res['eta_e']):.2f} dBi"
          if res['eta_e'] > 0 else "—")
c3.metric("Element 3-dB BW θ₃", f"{res['theta_3_element']:.2f}°")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("N (calculated)", f"{res['N_raw']:.1f}")
c2.metric("N (rounded up)", f"{res['N']}")
c3.metric("N (placed in grid)", f"{res['N_placed']}")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Peak Directivity (placed)", f"{res['Dp_placed_dBi']:.2f} dBi")
c2.metric("Directivity at scan edge", f"{res['Dp_placed_dBi'] - res['SL_dB']:.2f} dBi")
c3.metric("Aperture diameter",
          f"{res['diameter_circ_m']*1000:.1f} mm  ({res['diameter_circ_lambda']:.1f}λ)")

st.markdown("---")

c1, c2 = st.columns(2)
c1.metric("Boresight Grating Lobe", f"{res['boresight_GL_deg']:.2f}°")
c2.metric("Grating Lobe at scan edge", f"{res['scan_GL_deg']:.2f}°")

st.success(
    f"A hexagonal array of **{res['N_placed']} elements** with "
    f"**d = {res['d_hex_mm']:.3f} mm** ({res['d_hex_norm']:.3f}λ) spacing "
    f"fits inside a circular aperture of **{res['diameter_circ_m']*1000:.1f} mm** diameter."
)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION C — Layout & Patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Part C — Array Layout & Radiation Patterns")

col1, col2 = st.columns(2)
with col1:
    fig_layout = plot_array_layout(
        res["x_pos"], res["y_pos"], res["weights"], res["R_circ"],
        res["d_hex_m"],
        title=f"Hexagonal Array Layout  ({res['N_placed']} elements)")
    st.pyplot(fig_layout, use_container_width=True)

with col2:
    fig_eff = plot_efficiency_vs_taper()
    st.pyplot(fig_eff, use_container_width=True)

st.markdown("---")

st.markdown("#### Radiation Pattern (φ = 0° cut)")
fig_pat = plot_radiation_pattern(
    res["x_pos"], res["y_pos"], res["weights"],
    res["lambda_c"], res["Dp_placed_dBi"],
    title=f"Array Factor — {res['N_placed']} elements, {taper_dB:.0f} dB taper")
st.pyplot(fig_pat, use_container_width=True)

st.markdown("#### Normalized Pattern")
fig_norm = plot_normalized_pattern(
    res["x_pos"], res["y_pos"], res["weights"],
    res["lambda_c"], res["Dp_placed_dBi"],
    title="Normalized Array Radiation Pattern")
st.pyplot(fig_norm, use_container_width=True)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION D — Complexity Reduction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Part D — Complexity Reduction with Reduced Scan Angle")

theta_reduced = st.slider("Reduced Scan Angle (degrees)", min_value=1.0,
                          max_value=float(theta_sm), value=6.0, step=0.5)

res_red = run_design(f_center, f_offset, theta_reduced, G_min, taper_dB,
                     eta_e_pct, element_name, L_s, GL_pe, X, I_m)

st.markdown("---")
st.markdown("### Side-by-side Comparison")

compare_data = {
    "Parameter": [
        "Max Scan Angle (°)",
        "Grating Lobe θ_G (°)",
        "Element Spacing d/λ",
        "Element Spacing (mm)",
        "Element Directivity (dBi)",
        "Scan Loss (dB)",
        "Required D_p (dBi)",
        "Number of Elements (grid)",
        "Aperture Diameter (mm)",
        "Boresight GL (°)",
        "Scan-edge GL (°)",
    ],
    f"±{theta_sm:.0f}° (Original)": [
        f"{theta_sm:.1f}",
        f"{res['theta_G']:.1f}",
        f"{res['d_hex_norm']:.4f}",
        f"{res['d_hex_mm']:.3f}",
        f"{res['De_dBi']:.2f}",
        f"{res['SL_dB']:.2f}",
        f"{res['Dp_dBi']:.2f}",
        f"{res['N_placed']}",
        f"{res['diameter_circ_m']*1000:.1f}",
        f"{res['boresight_GL_deg']:.1f}",
        f"{res['scan_GL_deg']:.1f}",
    ],
    f"±{theta_reduced:.0f}° (Reduced)": [
        f"{theta_reduced:.1f}",
        f"{res_red['theta_G']:.1f}",
        f"{res_red['d_hex_norm']:.4f}",
        f"{res_red['d_hex_mm']:.3f}",
        f"{res_red['De_dBi']:.2f}",
        f"{res_red['SL_dB']:.2f}",
        f"{res_red['Dp_dBi']:.2f}",
        f"{res_red['N_placed']}",
        f"{res_red['diameter_circ_m']*1000:.1f}",
        f"{res_red['boresight_GL_deg']:.1f}",
        f"{res_red['scan_GL_deg']:.1f}",
    ],
}

st.dataframe(compare_data, use_container_width=True, hide_index=True)

st.markdown("---")

# Metrics
if res["N_placed"] > 0 and res_red["N_placed"] > 0:
    reduction_pct = (1 - res_red["N_placed"] / res["N_placed"]) * 100
    spacing_increase = (res_red["d_hex_norm"] / res["d_hex_norm"] - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Element Reduction",
              f"{abs(reduction_pct):.1f} %",
              delta=f"{res['N_placed'] - res_red['N_placed']} fewer elements",
              delta_color="normal" if reduction_pct > 0 else "inverse")
    c2.metric("Spacing Increase",
              f"{spacing_increase:.1f} %",
              delta=f"{res_red['d_hex_mm'] - res['d_hex_mm']:.3f} mm wider")
    c3.metric("Scan Loss Reduction",
              f"{res['SL_dB'] - res_red['SL_dB']:.2f} dB saved")

st.markdown("---")

st.markdown("### Reduced-Scan Array Layout")
col1, col2 = st.columns(2)
with col1:
    fig_red = plot_array_layout(
        res_red["x_pos"], res_red["y_pos"], res_red["weights"],
        res_red["R_circ"], res_red["d_hex_m"],
        title=f"±{theta_reduced:.0f}° Scan — {res_red['N_placed']} elements")
    st.pyplot(fig_red, use_container_width=True)
with col2:
    fig_pat_red = plot_radiation_pattern(
        res_red["x_pos"], res_red["y_pos"], res_red["weights"],
        res_red["lambda_c"], res_red["Dp_placed_dBi"],
        title=f"Pattern — ±{theta_reduced:.0f}° Scan")
    st.pyplot(fig_pat_red, use_container_width=True)

st.info(
    f"**Summary:** Reducing the scan angle from ±{theta_sm:.0f}° to ±{theta_reduced:.0f}° "
    f"allows larger element spacing ({res_red['d_hex_norm']:.3f}λ vs {res['d_hex_norm']:.3f}λ), "
    f"which increases element directivity and reduces the total element count from "
    f"**{res['N_placed']}** to **{res_red['N_placed']}** "
    f"(a **{abs(reduction_pct):.1f}%** reduction). "
    f"This translates directly to fewer phase shifters, a simpler feed network, "
    f"lower power consumption, reduced weight, and significant cost savings."
)

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION — Key Design Equations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("📚  Key Design Equations")
st.markdown(r"""
**Hexagonal lattice spacing** — *Eq. (2)*:

$$\frac{d_h}{\lambda_h} \leq \frac{1.1547}{\sin\theta_{sm} + \sin\theta_G}$$

**Array peak directivity** — *Eq. (3)*:

$$D_p = 10\log_{10}(N) + 10\log_{10}\!\left[\eta_e \frac{4\pi A_e}{\lambda_l^2}\right]$$

**Number of elements** — *Eq. (4)*:

$$N = 10^{\,0.1\,D_p\;-\;0.1\,D_e}$$

**Required peak directivity** — *Eq. (5)*:

$$D_p = G_{\min} + L_s + SL + GL_{pe} + T_L + X + I_m$$

**Scan loss (directive elements)** — *Eq. (6)*:

$$SL = 3\left(\frac{\theta_{sm}}{0.5\,\theta_3}\right)^2$$

**Element half-power beamwidth** — *Eq. (7)*:

$$\theta_3 = A\,\frac{\lambda_h}{d_e}$$

**Taper efficiency** — *Eq. (10)*:

$$\eta = 75\,\frac{(1+T)^2}{1+T+T^2}\;\;\%$$

---
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("CICAD 2025 Assignment — Phased Array Problem 2")
