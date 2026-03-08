import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import jv

# --- Functions for Antenna Calculations ---
def calculate_reflector_geometry(D, f_D):
    F = f_D * D
    # Subtended half-angle at the edge of the reflector
    theta_0_rad = 2 * math.atan(1 / (4 * f_D))
    theta_0_deg = math.degrees(theta_0_rad)
    return F, theta_0_deg

def calculate_feed_array(frequency_GHz, spacing_lambda, element_eff):
    c = 3e8
    f = frequency_GHz * 1e9
    wavelength = c / f
    
    # 2x2 Array physical size
    # Total effective size is roughly 2 * spacing_lambda
    array_size_lambda = 2 * spacing_lambda
    
    # Half-power beamwidth of the array feed (using patch element factor ~ 58 degrees)
    # HPBW ~ 58 / (Aperture in wavelengths)
    theta_3dB_feed = 58 / array_size_lambda
    
    # Directivity of 2x2 Array
    # De = 10 * log10( (4*pi / lambda^2) * physical_area * efficiency )
    # Unit cell area per element = spacing^2
    total_area_lambda2 = 4 * (spacing_lambda**2)
    directivity_linear = 4 * math.pi * total_area_lambda2 * element_eff
    directivity_dBi = 10 * math.log10(directivity_linear)
    
    return wavelength, theta_3dB_feed, directivity_dBi

def calculate_efficiencies(theta_0_deg, theta_3dB_feed):
    # Edge illumination taper in dB using Gaussian beam approximation
    T_dB = 12 * (theta_0_deg / theta_3dB_feed)**2
    
    # Spillover efficiency
    # eta_s = 1 - 10^(-T_dB / 10)
    eta_s = 1 - 10**(-T_dB / 10)
    
    # Illumination efficiency (approximate empirical formula for parabolic taper)
    # Alternatively using typical Gaussian beam mapping:
    T_linear = 10**(T_dB / 20)
    # Using classical approximation for illumination efficiency
    if T_dB > 0:
        eta_i = ( (1 - 10**(-T_dB/20))**2 ) / ( (T_dB/20) * math.log(10) )
    else:
        eta_i = 1.0
        
    eta_ap = eta_s * eta_i
    return T_dB, eta_s, eta_i, eta_ap

def calculate_secondary_beam(D, wavelength, eta_ap, T_dB):
    # Peak Directivity
    D_lambda = D / wavelength
    peak_directivity_linear = eta_ap * (math.pi * D_lambda)**2
    peak_directivity_dBi = 10 * math.log10(peak_directivity_linear)
    
    # 3dB Beamwidth (Secondary) - varies slightly with taper. 
    # Usually around 70 * (lambda / D) for practical tapers.
    beamwidth_factor = 58 + 0.8 * T_dB if T_dB < 20 else 74 # Empirical relation
    theta_3dB_sec = beamwidth_factor * (wavelength / D)
    
    return peak_directivity_dBi, theta_3dB_sec

def plot_secondary_pattern(D, wavelength, T_dB, peak_dir):
    theta = np.linspace(-3, 3, 1000)
    theta_rad = np.radians(theta)
    
    # Universal parameter u
    u = math.pi * (D / wavelength) * np.sin(theta_rad)
    
    # Uniform circular aperture pattern modified by taper (approximate with Bessel)
    # For a purely uniform aperture: E(u) = 2 * J1(u)/u
    # We apply a simulated sidelobe reduction based on taper
    # SLL approx = -17.6 dB for uniform, drops ~ 0.5-0.7 dB per dB of taper
    
    epsilon = 1e-9
    u = np.where(u == 0, epsilon, u)
    pattern = 20 * np.log10(np.abs(2 * jv(1, u) / u))
    
    # Adjust sidelobes empirically based on taper
    sll_reduction = T_dB * 0.7
    pattern_adjusted = np.where(pattern < -3, pattern - (sll_reduction * (np.abs(pattern)/20)), pattern)
    
    normalized_pattern = peak_dir + pattern_adjusted
    return theta, normalized_pattern


# --- STREAMLIT UI ---
st.set_page_config(page_title="Reflector & Phased Array Analyzer", layout="wide")
st.title("Center-Fed Reflector & Phased Array Analyzer")
st.markdown("Solution for CICAD 2025 Assignment Problem")

# Inputs Setup
st.sidebar.header("System Parameters")
D = st.sidebar.number_input("Reflector Diameter (m)", value=2.5)
f_D = st.sidebar.number_input("f/D Ratio", value=0.9)
freq = st.sidebar.number_input("Operating Frequency (GHz)", value=12.0)
grid_elements = st.sidebar.text("Feed Array: 2x2 Patch")
spacing = st.sidebar.number_input("Initial Element Spacing (lambda)", value=0.5)
eff_patch = st.sidebar.number_input("Patch Efficiency (%)", value=90.0) / 100.0

F, theta_0 = calculate_reflector_geometry(D, f_D)
wavelength, theta_3dB_feed, dir_feed = calculate_feed_array(freq, spacing, eff_patch)
T_dB, eta_s, eta_i, eta_ap = calculate_efficiencies(theta_0, theta_3dB_feed)
peak_dir_sec, theta_3dB_sec = calculate_secondary_beam(D, wavelength, eta_ap, T_dB)

st.divider()

# --- Part A: Geometry Sketch ---
st.header("Part A: Reflector and Feed Configuration Sketch")
fig_a, ax_a = plt.subplots(figsize=(6, 4))
y = np.linspace(-D/2, D/2, 100)
x = (y**2) / (4*F) # Parabola equation

ax_a.plot(x, y, 'b', linewidth=2, label="Reflector")
ax_a.plot(F, 0, 'ro', markersize=8, label="2x2 Array Feed")
ax_a.plot([F, 0], [0, D/2], 'k--', alpha=0.5)
ax_a.plot([F, 0], [0, -D/2], 'k--', alpha=0.5)

ax_a.annotate(f"D = {D}m", xy=(0, D/2), xytext=(-0.5, D/2.2), arrowprops=dict(arrowstyle="->"))
ax_a.annotate(f"F = {F:.2f}m", xy=(F/2, 0), xytext=(F/2, -0.3))
ax_a.annotate(rf"$\theta_0$ = {theta_0:.1f}°", xy=(F, 0), xytext=(F-0.4, 0.2))

ax_a.set_xlim(-0.2, F + 0.5)
ax_a.set_ylim(-D/2 - 0.2, D/2 + 0.2)
ax_a.set_xlabel("Optical Axis (m)")
ax_a.set_ylabel("Aperture (m)")
ax_a.set_title("Center Fed Parabolic Reflector")
ax_a.legend()
ax_a.grid(True)
st.pyplot(fig_a)

# --- Part B & C & D: Calculations ---
col1, col2 = st.columns(2)
with col1:
    st.header("Part B: Phased Array Feed")
    st.write(f"**Element Spacing:** {spacing} $\lambda$")
    st.write(f"**Array Directivity:** {dir_feed:.2f} dBi")
    st.write(f"**Array 3-dB Beamwidth:** {theta_3dB_feed:.2f}°")
    st.write(f"**Illumination Taper at Edges ($T$):** {T_dB:.2f} dB")
    
with col2:
    st.header("Part C & D: Reflector Secondary Beam")
    st.write(f"**Spillover Efficiency:** {eta_s*100:.1f}%")
    st.write(f"**Illumination Efficiency:** {eta_i*100:.1f}%")
    st.write(f"**Total Aperture Efficiency:** {eta_ap*100:.1f}%")
    st.write(f"**Peak Secondary Directivity:** {peak_dir_sec:.2f} dBi")
    st.write(f"**Secondary 3-dB Beamwidth:** {theta_3dB_sec:.3f}°")

st.divider()

# --- Part E: Secondary Beam Patterns ---
st.header("Part E: Reflector Secondary Beam Pattern")

theta_angles, pattern = plot_secondary_pattern(D, wavelength, T_dB, peak_dir_sec)

# Find first side lobe level (Approximate logic)
peaks, _ = scipy.signal.find_peaks(pattern) if 'scipy.signal' in sys.modules else ([], [])
# Fallback to standard analytical uniform circular aperture FSLL - adjusted for taper
fsll_relative = -17.6 - (T_dB * 0.5)
fsll_absolute = peak_dir_sec + fsll_relative

fig_e, ax_e = plt.subplots(figsize=(8, 4))
ax_e.plot(theta_angles, pattern, 'b-', label='Secondary Pattern')
ax_e.axhline(peak_dir_sec - 3, color='r', linestyle='--', label='3 dB Beamwidth limit')
ax_e.axhline(fsll_absolute, color='g', linestyle='--', label=f'First SLL (~{fsll_relative:.1f} dBc)')
ax_e.set_xlim(-1.5, 1.5)
ax_e.set_ylim(max(0, peak_dir_sec - 50), peak_dir_sec + 2)
ax_e.set_xlabel("Theta (degrees)")
ax_e.set_ylabel("Directivity (dBi)")
ax_e.set_title("Approximate Reflector Secondary Pattern")
ax_e.grid(True)
ax_e.legend()
st.pyplot(fig_e)

st.markdown("""
**Suggestions to reduce the First Side Lobe Level (FSLL):**
1. **Increase the Illumination Taper:** By increasing the taper (e.g., from ~3.5 dB to ~10-12 dB), less power hits the edges of the reflector, smoothing the aperture field distribution and drastically lowering sidelobes.
2. **Increase Feed Array Spacing or Size:** Making the feed array physically larger (e.g., increasing spacing to $0.7\lambda$) makes the primary beam narrower, naturally increasing the edge taper on the reflector.
""")

st.divider()

# --- Part F: Replacing Element Spacing ---
st.header("Part F: Changing Element Spacing to 0.7 $\lambda$")

spacing_new = 0.7
_, theta_3dB_feed_new, dir_feed_new = calculate_feed_array(freq, spacing_new, eff_patch)
T_dB_new, eta_s_new, eta_i_new, eta_ap_new = calculate_efficiencies(theta_0, theta_3dB_feed_new)
peak_dir_sec_new, theta_3dB_sec_new = calculate_secondary_beam(D, wavelength, eta_ap_new, T_dB_new)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Original (0.5 $\lambda$)")
    st.write(f"- **Feed HPBW:** {theta_3dB_feed:.1f}°")
    st.write(f"- **Edge Taper:** {T_dB:.2f} dB")
    st.write(f"- **Aperture Eff:** {eta_ap*100:.1f}%")
    st.write(f"- **Secondary Peak Directivity:** {peak_dir_sec:.2f} dBi")
with col4:
    st.subheader("New (0.7 $\lambda$)")
    st.write(f"- **Feed HPBW:** {theta_3dB_feed_new:.1f}°")
    st.write(f"- **Edge Taper:** {T_dB_new:.2f} dB")
    st.write(f"- **Aperture Eff:** {eta_ap_new*100:.1f}%")
    st.write(f"- **Secondary Peak Directivity:** {peak_dir_sec_new:.2f} dBi")

st.markdown("""
**Conceptual Analysis:**
When element spacing increases from 0.5 $\lambda$ to 0.7 $\lambda$, the physical aperture of the feed array increases.
* **Feed Array Beamwidth (Primary Beam):** Decreases (narrows).
* **Illumination Taper at Edge:** **Increases**. A narrower feed beam means less power hits the edges of the reflector.
* **Spillover Efficiency:** **Increases** because less energy is spilling past the reflector edges.
* **Illumination Efficiency:** **Decreases** because the reflector is now under-illuminated (the energy is concentrated too much in the center).
* **First Side Lobe Level (Secondary Beam):** **Decreases (Improves)** due to the heavier amplitude taper across the reflector aperture.
* **Secondary Beamwidth:** Will **increase slightly** as the effective radiating area of the reflector shrinks towards the center.
""")
