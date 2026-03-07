import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

def calculate_array_params(f_center, f_margin, scan_angle, gain_min, taper_db, eff_horn):
    # Frequencies in GHz -> Wavelengths in meters
    c = 0.299792458
    f_high = f_center + f_margin
    f_low = f_center - f_margin
    lambda_h = c / f_high
    lambda_l = c / f_low
    
    # 1. Element Spacing (Hexagonal) using highest frequency
    # Grating lobe placed 1 degree outside scan region for small scans
    grating_lobe = scan_angle + 1.0 
    
    denom = math.sin(math.radians(scan_angle)) + math.sin(math.radians(grating_lobe))
    spacing_lambda_h = 1.1547 / denom
    spacing_meters = spacing_lambda_h * lambda_h
    
    # Spacing in terms of lowest frequency wavelength for directivity
    spacing_lambda_l = spacing_meters / lambda_l
    
    # 2. Element Directivity
    # Hexagonal unit cell area A = (sqrt(3)/2) * d^2, but based on context equations 
    # we approximate area to spacing^2 for the directivity formula consistency.
    element_eff = eff_horn / 100.0
    d_e = 10 * math.log10(element_eff * 4 * math.pi * (spacing_lambda_l**2))
    
    # 3. Peak Directivity Required
    # Taper loss calculation
    T = 10**(taper_db / 20.0)
    eff_taper = 75 * ((1 + T)**2 / (1 + T + T**2)) / 100.0
    taper_loss = -10 * math.log10(eff_taper)
    
    # Scan loss calculation (Potter horn constant A = 70)
    theta_3 = 70.0 * (1.0 / spacing_lambda_h)
    scan_loss = 3 * (scan_angle / (0.5 * theta_3))**2
    
    peak_directivity = gain_min + scan_loss + taper_loss
    
    # 4. Number of Elements
    num_elements = 10**(0.1 * peak_directivity - 0.1 * d_e)
    
    return {
        "spacing_lambda_h": spacing_lambda_h,
        "spacing_meters": spacing_meters,
        "element_directivity": d_e,
        "taper_loss": taper_loss,
        "scan_loss": scan_loss,
        "peak_directivity": peak_directivity,
        "num_elements": math.ceil(num_elements)
    }

def plot_hexagonal_lattice(num_elements, spacing_in, plot_title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.set_title(plot_title)
    ax.set_xlabel("Element Spacing (in)")
    ax.set_ylabel("Element Spacing (in)")
    
    # Estimate depth based on hexagonal rings
    depth = int(math.ceil(math.sqrt(num_elements / 3.0)))
    ax.set_ylim([-depth * spacing_in, depth * spacing_in])
    ax.set_xlim([-depth * spacing_in, depth * spacing_in])
    
    points_x, points_y = [0], [0]
    for q in range(-depth, depth + 1):
        for r in range(max(-depth, -q - depth), min(depth, -q + depth) + 1):
            if q == 0 and r == 0:
                continue
            x = spacing_in * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
            y = spacing_in * (3/2 * r)
            points_x.append(x)
            points_y.append(y)
            
    # Scatter and draw circles for the exact number of elements
    points_x = points_x[:num_elements]
    points_y = points_y[:num_elements]
    
    for i in range(len(points_x)):
        circle = plt.Circle((points_x[i], points_y[i]), radius=spacing_in/2, fill=True, color='#1f77b4', alpha=0.8)
        ax.add_patch(circle)
        
    ax.scatter(points_x, points_y, color='black', s=5)
    return fig

def plot_bessel_pattern(D_p, L_hx):
    plot_angle = 30
    in_array = np.linspace(-plot_angle, plot_angle, 1000)
    # Avoid zero division
    in_array[in_array == 0] = 1e-5 
    
    t = np.pi * (in_array / 180.0)
    bessel_trail = np.pi * L_hx * np.sin(t)
    bessel_pattern = D_p + 10 * np.log10((2 * jv(1, bessel_trail) / bessel_trail)**2)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(in_array, bessel_pattern)
    ax.set_xlabel('Theta (Degrees)')
    ax.set_ylabel('Directivity (dBi)')
    ax.set_title('Circular Aperture Radiation Pattern (Bessel)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim([D_p - 40, D_p + 5])
    return fig

st.set_page_config(page_title="Phased Array Designer", layout="centered")
st.title("CICAD 2025: Phased Array Problem 2")

st.sidebar.header("Design Specifications")
f_center = st.sidebar.number_input("Center Frequency (GHz)", value=44.5)
f_margin = st.sidebar.number_input("Freq Margin +/- (GHz)", value=1.0)
scan_angle = st.sidebar.number_input("Max Scan Angle (deg)", value=9.0)
gain_min = st.sidebar.number_input("Min Gain over Scan (dBi)", value=40.0)
taper_db = st.sidebar.number_input("Amplitude Taper (dB)", value=10.0)
eff_horn = st.sidebar.number_input("Horn Efficiency (%)", value=70.0)

if st.button("Calculate & Plot"):
    res = calculate_array_params(f_center, f_margin, scan_angle, gain_min, taper_db, eff_horn)
    
    st.subheader("Calculated Parameters")
    col1, col2 = st.columns(2)
    col1.metric("Spacing (\u03bb_high)", f"{res['spacing_lambda_h']:.2f}")
    col1.metric("Element Directivity", f"{res['element_directivity']:.2f} dBi")
    col2.metric("Required Peak Directivity", f"{res['peak_directivity']:.2f} dBi")
    col2.metric("Number of Elements", res['num_elements'])
    
    st.markdown("---")
    st.subheader("Visualizations")
    
    # Calculate dimensional footprint for plotting
    meters_to_in = 39.37
    spacing_in = res['spacing_meters'] * meters_to_in
    num_x = math.ceil(math.sqrt(res['num_elements']))
    L_hx = num_x * res['spacing_lambda_h']
    
    fig_layout = plot_hexagonal_lattice(res['num_elements'], spacing_in, "Hexagonal Array Layout")
    st.pyplot(fig_layout)
    
    fig_pattern = plot_bessel_pattern(res['peak_directivity'], L_hx)
    st.pyplot(fig_pattern)
