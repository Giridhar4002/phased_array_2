import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

def calculate_array_params(f_center, f_margin, scan_angle, theta_g, gain_min, taper_db, eff_horn, misc_losses):
    # Frequencies in GHz -> Wavelengths in meters
    c = 0.299792458
    f_high = f_center + f_margin
    f_low = f_center - f_margin
    lambda_h = c / f_high
    lambda_l = c / f_low
    
    # 1. Element Spacing (Hexagonal) using highest frequency to avoid grating lobes
    denom = math.sin(math.radians(scan_angle)) + math.sin(math.radians(theta_g))
    spacing_lambda_h = 1.1547 / denom
    spacing_meters = spacing_lambda_h * lambda_h
    
    # Spacing in terms of lowest frequency wavelength for directivity
    spacing_lambda_l = spacing_meters / lambda_l
    
    # 2. Element Directivity (De)
    element_eff = eff_horn / 100.0
    d_e = 10 * math.log10(element_eff * 4 * math.pi * (spacing_lambda_l**2))
    
    # 3. Peak Directivity Required (Dp)
    # Taper efficiency calculation
    T = 10**(-taper_db / 20.0)
    eff_taper = 0.75 * ((1 + T)**2 / (1 + T + T**2))
    taper_loss = -10 * math.log10(eff_taper)
    
    # Scan loss calculation (Potter horn constant A = 70)
    theta_3 = 70.0 / spacing_lambda_h
    scan_loss = 3 * (scan_angle / (0.5 * theta_3))**2
    
    # Total required peak directivity
    peak_directivity = gain_min + taper_loss + misc_losses + scan_loss
    
    # 4. Number of Elements (Raw theoretical)
    num_elements_raw = 10**(0.1 * peak_directivity - 0.1 * d_e)
    
    # 5. Aperture Diameter (D_aperture)
    # Area of hex unit cell = (sqrt(3)/2) * dh^2. 
    # For a circular aperture, Area = pi * (D/2)^2 = N * Area_cell
    D_aperture = math.sqrt(4 * num_elements_raw / (math.pi * 1.1547)) * spacing_meters
    
    return {
        "lambda_h": lambda_h,
        "spacing_lambda_h": spacing_lambda_h,
        "spacing_meters": spacing_meters,
        "element_directivity": d_e,
        "taper_loss": taper_loss,
        "scan_loss": scan_loss,
        "peak_directivity": peak_directivity,
        "num_elements_raw": num_elements_raw,
        "D_aperture": D_aperture
    }

def plot_hexagonal_lattice_circular(D_aperture, spacing_m, plot_title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_title(plot_title)
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    
    radius_m = D_aperture / 2.0
    spacing_mm = spacing_m * 1000.0
    radius_mm = radius_m * 1000.0
    
    # Estimate max indices to cover the circle
    max_idx = int(math.ceil(radius_mm / spacing_mm)) * 2
    ax.set_ylim([-radius_mm * 1.2, radius_mm * 1.2])
    ax.set_xlim([-radius_mm * 1.2, radius_mm * 1.2])
    
    points_x, points_y = [], []
    
    # Generate hexagonal grid points
    for q in range(-max_idx, max_idx + 1):
        for r in range(-max_idx, max_idx + 1):
            x = spacing_mm * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
            y = spacing_mm * (3/2 * r)
            
            # Clip elements outside the circular aperture boundary
            if math.sqrt(x**2 + y**2) <= radius_mm:
                points_x.append(x)
                points_y.append(y)
                
    actual_elements = len(points_x)
    
    # Draw elements
    for i in range(actual_elements):
        circle = plt.Circle((points_x[i], points_y[i]), radius=spacing_mm/2 * 0.9, fill=True, color='#1f77b4', alpha=0.8)
        ax.add_patch(circle)
        
    ax.scatter(points_x, points_y, color='black', s=5, zorder=3)
    
    # Draw the theoretical circular aperture boundary
    aperture_circle = plt.Circle((0, 0), radius=radius_mm, fill=False, color='red', linestyle='--', linewidth=2, label='Aperture Boundary')
    ax.add_patch(aperture_circle)
    ax.legend(loc="upper right")
    
    return fig, actual_elements

def plot_bessel_pattern(D_p, D_aperture, lambda_h):
    plot_angle = 30
    theta_deg = np.linspace(-plot_angle, plot_angle, 1000)
    theta_deg[theta_deg == 0] = 1e-5  # Avoid zero division
    
    theta_rad = np.deg2rad(theta_deg)
    
    # u = pi * (D / lambda) * sin(theta)
    u = np.pi * (D_aperture / lambda_h) * np.sin(theta_rad)
    
    # f(theta) = Dp + 20*log10(|2*J1(u)/u|)
    bessel_factor = 20 * np.log10(np.abs(2 * jv(1, u) / u))
    bessel_pattern = D_p + bessel_factor
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(theta_deg, bessel_pattern, color='indigo')
    ax.set_xlabel('Theta (Degrees)')
    ax.set_ylabel('Directivity (dBi)')
    ax.set_title('Circular Aperture Radiation Pattern (Bessel)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim([D_p - 40, D_p + 5])
    return fig

st.set_page_config(page_title="Phased Array Designer", layout="centered")
st.title("CICAD 2025: Phased Array Problem 2")
st.markdown("Calculates array parameters based on Potter Horn elements in a circular-aperture hexagonal lattice.")

st.sidebar.header("Design Specifications")
f_center = st.sidebar.number_input("Center Frequency (GHz)", value=44.5, step=0.1)
f_margin = st.sidebar.number_input("Freq Margin +/- (GHz)", value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Scan & Constraints")
scan_angle = st.sidebar.number_input("Max Scan Angle (deg)", value=9.0, step=1.0)
theta_g = st.sidebar.number_input("Grating Lobe Margin Angle (deg)", value=9.7, step=0.1, help="Use 9.7 for 9° scan, or 7.0 for 6° scan per the text.")

gain_min = st.sidebar.number_input("Min Gain over Scan (dBi)", value=40.0, step=1.0)
misc_losses = st.sidebar.number_input("Misc Losses (dB)", value=4.0, step=0.1, help="Sum of Ls, GLpe, Im, X (e.g. 0.5 + 0 + 0.5 + 3.0 = 4.0)")

st.sidebar.markdown("---")
st.sidebar.subheader("Element & Aperture")
taper_db = st.sidebar.number_input("Amplitude Taper (dB)", value=10.0, step=1.0)
eff_horn = st.sidebar.number_input("Horn Efficiency (%)", value=70.0, step=1.0)

if st.button("Calculate & Plot"):
    res = calculate_array_params(f_center, f_margin, scan_angle, theta_g, gain_min, taper_db, eff_horn, misc_losses)
    
    st.subheader("Calculated Parameters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Spacing (d_h)", f"{res['spacing_lambda_h']:.3f} \u03bb_h")
    col1.metric("Element Directivity", f"{res['element_directivity']:.2f} dBi")
    
    col2.metric("Scan Loss", f"{res['scan_loss']:.2f} dB")
    col2.metric("Required Peak Dir.", f"{res['peak_directivity']:.2f} dBi")
    
    col3.metric("Theoretical Elements", f"{res['num_elements_raw']:.1f}")
    col3.metric("Aperture Diameter", f"{res['D_aperture']*1000:.1f} mm")
    
    st.markdown("---")
    st.subheader("Visualizations")
    
    # 1. Hexagonal Array Layout (Circular Aperture)
    fig_layout, actual_count = plot_hexagonal_lattice_circular(res['D_aperture'], res['spacing_meters'], "Hexagonal Array Layout")
    st.pyplot(fig_layout)
    st.caption(f"**Actual Elements Fitted in Circular Aperture:** {actual_count}")
    
    # 2. Bessel Pattern
    fig_pattern = plot_bessel_pattern(res['peak_directivity'], res['D_aperture'], res['lambda_h'])
    st.pyplot(fig_pattern)
