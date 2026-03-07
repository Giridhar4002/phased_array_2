
"""
=============================================================
 PHASED ARRAY ANTENNA DESIGN — 44.5 GHz, Hexagonal Grid
=============================================================
 Specs:
   • Frequency : 44.5 GHz ± 1 GHz
   • Grid : Hexagonal
   • Aperture : Circular (Bessel pattern)
   • Feed Element : Potter Horn (aperture eff = 70 %)
   • Scan Region : ± 9 ° (Part d also solves ± 6 °)
   • Min Gain : 40 dBi over coverage region
   • Amplitude Taper : 10 dB across aperture
=============================================================
"""
"""
Phased Array Antenna Design — Streamlit App
44.5 GHz, Hexagonal Grid, Potter Horn, ±9° scan, 40 dBi min gain, 10 dB taper
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.special import jv
# ──────────────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────────────
st.set_page_config(page_title="Phased Array Antenna Design", page_icon="📡", layout="wide")
st.title("📡 Phased Array Antenna Design — 44.5 GHz Hexagonal Array")
st.markdown("**Hexagonal grid · Circular aperture · Potter Horn (70% eff.) · ±9° scan · Min gain 40 dBi · 10 dB amplitude taper**")
# ──────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ──────────────────────────────────────────────────────
st.sidebar.header("Input Parameters")
freq_ghz = st.sidebar.number_input("Frequency (GHz)", min_value=1.0, max_value=100.0, value=44.5, step=0.5)
freq_bw = st.sidebar.number_input("Bandwidth +/- (GHz)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
scan_main = st.sidebar.number_input("Scan Angle - Parts a-c (deg)", min_value=1.0, max_value=60.0, value=9.0, step=1.0)
scan_red = st.sidebar.number_input("Reduced Scan Angle - Part d (deg)", min_value=1.0, max_value=60.0, value=6.0, step=1.0)
gain_min = st.sidebar.number_input("Min Gain over Coverage (dBi)", min_value=10.0, max_value=70.0, value=40.0, step=1.0)
elem_eff = st.sidebar.number_input("Element Efficiency (%)", min_value=10.0, max_value=100.0, value=70.0, step=5.0)
taper_db = st.sidebar.number_input("Amplitude Taper (dB)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
horn_k = st.sidebar.number_input("Potter Horn factor k (deg)", min_value=50, max_value=90, value=70, step=1)
run = st.sidebar.button("Run Design", use_container_width=True)
# ──────────────────────────────────────────────────────
# PHYSICS FUNCTIONS
# ──────────────────────────────────────────────────────
C = 0.299792458 # GHz*m
hex_factor = math.sqrt(3) / 2
def array_efficiency(taper):
    T = 10 ** (-taper / 20.0)
    return 75.0 * (1 + T) ** 2 / (1 + T + T ** 2)
def grating_lobe_loc(scan, t3db):
    t3s = t3db / math.sqrt(math.cos(math.radians(scan)) ** 1.2)
    gl = scan + 1.5 * t3s
    return min(gl, 90.0), gl > 90.0
def hex_spacing(scan, gl):
    return 1.1547 / (math.sin(math.radians(scan)) + 1.0)
def get_scan_loss(scan, d, k):
    if d < 1.0:
        return -10.0 * math.log10(math.cos(math.radians(scan)) ** 1.5)
    return 3.0 * (scan / (0.5 * k / d)) ** 2
def elem_directivity(eff, d):
    return 10.0 * math.log10(0.01 * eff * 4 * math.pi * d ** 2 * hex_factor)
def design(scan, freq, gmin, eff, taper, k):
    wl = C / freq
    wl_mm = wl * 1000
    ae = array_efficiency(taper)
    dbore = gmin / (ae * 0.01)
    Dlin = (10 ** (dbore / 10)) / 0.9
    Dm = math.sqrt(Dlin) / math.pi * wl
    t3db = k * (wl / Dm)
    gl, lim = grating_lobe_loc(scan, t3db)
    d = hex_spacing(scan, gl)
    sl = get_scan_loss(scan, d, k)
    dir_req = (gmin + sl) / (ae * 0.01)
    ed = elem_directivity(eff, d)
    N = math.ceil(10 ** (0.1 * dir_req - 0.1 * ed))
    Dp = 10 * math.log10(0.01 * ae * N * 0.01 * eff * 4 * math.pi * d ** 2 * hex_factor)
    Ds = Dp - sl
    glb = math.degrees(math.asin(min(1.1547 / d, 1.0)))
    arg = 1.1547 / d - math.sin(math.radians(scan))
    gls = math.degrees(math.asin(arg)) if abs(arg) <= 1 else 90.0
    return dict(ae=ae, d=d, d_mm=d*wl_mm, d_in=d*wl*39.37,
                ed=ed, sl=sl, dir_req=dir_req, N=N, Dp=Dp, Ds=Ds,
                glb=glb, gls=gls, t3db=t3db, gl=gl, lim=lim,
                dbore=dbore, Dm_cm=Dm*100, wl_mm=wl_mm)
# ──────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ──────────────────────────────────────────────────────
def hexit_60(n_max):
    pairs = []
    if n_max >= 0:
        pairs.append(np.zeros((2, 1), dtype=int))
    if n_max >= 1:
        seq = [1, 0, -1]
        p0 = np.array(seq + seq[::-1])
        p1 = np.roll(p0, -2)
        pairs.append(np.stack((p0, p1), axis=0))
    for n in range(2, n_max + 1):
        seq = np.arange(n, -n - 1, -1, dtype=int)
        p0 = np.hstack((seq, (n - 1) * [-n], seq[::-1], (n - 1) * [n]))
        p1 = np.roll(p0, -2 * n)
        pairs.append(np.stack((p0, p1), axis=0))
    return np.hstack(pairs) if pairs else None
def hex_points(a, n_max):
    vecs = a * np.array([[1.0, 0.0], [0.5, 0.5 * np.sqrt(3)]])
    pairs = hexit_60(n_max)
    return (pairs[:, None, :] * vecs[:, :, None]).sum(axis=0)
def plot_layout(N, spacing_mm, title):
    depth, total = 0, 1
    for i in range(1, N + 1):
        total += 6 * i
        depth += 1
        if total >= N:
            break
    pts = hex_points(spacing_mm, depth - 1)
    x, y = pts
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.scatter(x, y, s=12, color="steelblue", zorder=3)
    for xi, yi in zip(x, y):
        ax.add_patch(plt.Circle((xi, yi), spacing_mm / 2,
                                alpha=0.2, color="steelblue", linewidth=0))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
def plot_patterns(res, scan, label, gmin):
    N = res["N"]; d = res["d"]; Dp = res["Dp"]
    Nx = math.ceil(math.sqrt(N))
    L = Nx * d
    ang = np.linspace(-50, 50, 2001)
    t = math.pi * ang / 180.0
    sinc_p = Dp + 10 * np.log10(np.sinc(L * t) ** 2 + 1e-30)
    u = math.pi * L * np.sin(t) + 1e-30
    bess_p = Dp + 10 * np.log10((2 * jv(1, u) / u) ** 2 + 1e-30)
    floor = Dp - 55
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pat, lbl in zip(axes, [sinc_p, bess_p],
                             ["Square Aperture (sinc2)", "Circular Aperture (Bessel2)"]):
        ax.plot(ang, np.clip(pat, floor, None), linewidth=1.5, color="#1f77b4")
        ax.axvline( scan, color="red", linestyle="--", lw=1.2, label=f"+/-{scan} deg scan")
        ax.axvline(-scan, color="red", linestyle="--", lw=1.2)
        ax.axhline(gmin, color="green", linestyle=":", lw=1.3, label=f"{gmin} dBi min")
        ax.set_xlabel("Theta (deg)"); ax.set_ylabel("Directivity (dBi)")
        ax.set_title(f"{lbl}\n{label}")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim([floor, Dp + 4])
    fig.tight_layout()
    return fig
def plot_n_vs_dir(res, scan, label, gmin, eff):
    N = res["N"]; d = res["d"]; sl = res["sl"]
    Nv = np.linspace(N * 0.5, N * 1.5, 200)
    ed_lin = 0.01 * eff * 4 * math.pi * d ** 2 * hex_factor
    Db = 10 * np.log10(Nv) + 10 * np.log10(ed_lin)
    Ds = Db - sl
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Nv, Db, label="Boresight")
    ax.plot(Nv, Ds, "--", label=f"{scan} deg Scan")
    ax.axhline(gmin, color="r", linestyle=":", label=f"{gmin} dBi min")
    ax.axvline(N, color="k", linestyle=":", label=f"N = {N}")
    ax.set_xlabel("Number of Elements"); ax.set_ylabel("Directivity (dBi)")
    ax.set_title(f"N vs Directivity - {label} (d/lam = {d:.3f})")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
def plot_dir_gl(res, scan, label, eff, k):
    d = res["d"]; N = res["N"]
    xs = np.linspace(0.7 * d, 1.3 * d, 300)
    ed_lin = 0.01 * eff * 4 * math.pi * xs ** 2 * hex_factor
    Db = 10 * np.log10(N) + 10 * np.log10(ed_lin)
    mask = xs >= 1.0
    sl = np.zeros_like(xs)
    sl[mask] = 3.0 * (scan / (0.5 * k / xs[mask])) ** 2
    sl[~mask] = -10.0 * np.log10(np.cos(np.radians(scan)) ** 1.5)
    Ds = Db - sl
    glb = np.where(xs > 1.1547, np.degrees(np.arcsin(np.clip(1.1547 / xs, -1, 1))), 90.0)
    arg = 1.1547 / xs - math.sin(math.radians(scan))
    gls = np.where(np.abs(arg) <= 1.0, np.degrees(np.arcsin(np.clip(arg, -1, 1))), 90.0)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    c1 = "#1f77b4"; c2 = "#d62728"
    ax1.plot(xs, Db, color=c1, label="Boresight")
    ax1.plot(xs, Ds, "--", color=c1, label=f"{scan} deg")
    ax1.set_xlabel("Element Spacing (d/lam)"); ax1.set_ylabel("Directivity (dBi)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.legend(loc="upper left", fontsize=8); ax1.grid(linestyle="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, glb, color=c2, label="GL boresight")
    ax2.plot(xs, gls, "--", color=c2, label=f"GL @ {scan} deg")
    ax2.set_ylabel("Grating-Lobe (deg)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"Directivity & Grating-Lobe vs Spacing - {label}")
    fig.tight_layout()
    return fig
def plot_bandwidth(res, freq, bw):
    wl = C / freq
    freqs = np.linspace(freq - bw, freq + bw, 100)
    wls = C / freqs
    d_phys= res["d"] * wl
    d_norm= d_phys / wls
    arg = 1.1547 / d_norm
    gl = np.where(arg <= 1.0, np.degrees(np.arcsin(arg)), 90.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, gl, linewidth=2)
    ax.axvline(freq, color="k", linestyle="--", label=f"Centre {freq} GHz")
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Grating-Lobe (deg)")
    ax.set_title("Grating-Lobe vs Frequency - Bandwidth Sensitivity")
    ax.legend(fontsize=9); ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
def plot_comparison(r9, r6, s9, s6):
    labels = [f"+/-{s9} deg", f"+/-{s6} deg"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#2196F3", "#4CAF50"]
    for ax, vals, ylabel, title in zip(
            axes,
            [[r9["N"], r6["N"]], [r9["Dp"], r6["Dp"]], [r9["d_mm"], r6["d_mm"]]],
            ["Number of Elements", "Peak Directivity (dBi)", "Element Spacing (mm)"],
            ["Element Count", "Peak Directivity", "Element Spacing"]):
        bars = ax.bar(labels, vals, color=colors, width=0.4)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    reduction = 100.0 * (r9["N"] - r6["N"]) / r9["N"]
    fig.suptitle(
        f"Part d - Scan Reduction +/-{s9} deg to +/-{s6} deg\n"
        f"Elements: {r9['N']} to {r6['N']} ({reduction:.1f}% reduction)",
        fontsize=11)
    fig.tight_layout()
    return fig
# ──────────────────────────────────────────────────────
# DEFAULT LANDING PAGE (shown before button press)
# ──────────────────────────────────────────────────────
if not run:
    st.info("Set parameters in the sidebar and click **Run Design** to generate full results.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequency", "44.5 GHz")
    c2.metric("Scan Angle", "+/- 9 deg")
    c3.metric("Min Gain", "40 dBi")
    c1.metric("Grid", "Hexagonal")
    c2.metric("Feed Element", "Potter Horn")
    c3.metric("Taper", "10 dB")
    st.stop()
# ──────────────────────────────────────────────────────
# RUN DESIGN
# ──────────────────────────────────────────────────────
with st.spinner("Calculating..."):
    r9 = design(scan_main, freq_ghz, gain_min, elem_eff, taper_db, horn_k)
    r6 = design(scan_red, freq_ghz, gain_min, elem_eff, taper_db, horn_k)
reduction_pct = 100.0 * (r9["N"] - r6["N"]) / r9["N"]
spacing_inc_pct = 100.0 * (r6["d"] - r9["d"]) / r9["d"]
# Top metrics row
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Wavelength", f"{r9['wl_mm']:.3f} mm")
c2.metric("Array Efficiency", f"{r9['ae']:.1f} %")
c3.metric("Elements Required", str(r9["N"]))
c4.metric("Peak Directivity", f"{r9['Dp']:.2f} dBi")
c5.metric(f"Dir. at scan edge", f"{r9['Ds']:.2f} dBi")
# ─── PART a ──────────────────────────────────────────
st.markdown("---")
st.header("Part a - Required Array Peak Directivity")
ca, cb = st.columns([1.3, 1])
with ca:
    df_a = pd.DataFrame({
        "Parameter": [
            "Frequency",
            "Wavelength",
            "Amplitude Taper",
            "Array Efficiency (10 dB taper)",
            "Estimated Aperture Diameter",
            "Boresight HPBW",
            "First Grating-Lobe Location",
            "Required Peak Directivity",
        ],
        "Value": [
            f"{freq_ghz} GHz",
            f"{r9['wl_mm']:.4f} mm",
            f"{taper_db} dB",
            f"{r9['ae']:.2f} %",
            f"{r9['Dm_cm']:.2f} cm",
            f"{r9['t3db']:.3f} deg",
            f"{r9['gl']:.2f} deg" + (" (limited to 90 deg)" if r9["lim"] else ""),
            f"{r9['dbore']:.2f} dBi",
        ]
    })
    st.dataframe(df_a, use_container_width=True, hide_index=True)
with cb:
    st.info(
        f"Min gain = {gain_min} dBi\n\n"
        f"Amplitude taper = {taper_db} dB\n\n"
        f"Array efficiency = {r9['ae']:.1f}%\n\n"
        f"First grating lobe placed at {r9['gl']:.1f} deg — just beyond the +/-{scan_main} deg scan edge.\n\n"
        f"**Required Peak Directivity = {r9['dbore']:.2f} dBi**"
    )
# ─── PART b ──────────────────────────────────────────
st.markdown("---")
st.header("Part b - Element Spacing, Element Gain & Number of Elements")
cb1, cb2 = st.columns([1.3, 1])
with cb1:
    df_b = pd.DataFrame({
        "Parameter": [
            "Element Spacing (d/lam)",
            "Element Spacing (mm)",
            "Element Spacing (inches)",
            "Element Directivity (dBi)",
            f"Scan Loss at +/-{scan_main} deg",
            "Required Directivity at Scan Edge",
            "Number of Elements (ceiling)",
            "Achieved Peak Directivity",
            f"Directivity at +/-{scan_main} deg",
            "Grating-Lobe at Boresight",
            "Grating-Lobe at Scan Edge",
        ],
        "Value": [
            f"{r9['d']:.4f} lam",
            f"{r9['d_mm']:.3f} mm",
            f"{r9['d_in']:.4f} in",
            f"{r9['ed']:.2f} dBi",
            f"{r9['sl']:.3f} dB",
            f"{r9['dir_req']:.2f} dBi",
            str(r9["N"]),
            f"{r9['Dp']:.2f} dBi",
            f"{r9['Ds']:.2f} dBi (req >= {gain_min} dBi)",
            f"{r9['glb']:.2f} deg",
            f"{r9['gls']:.2f} deg",
        ]
    })
    st.dataframe(df_b, use_container_width=True, hide_index=True)
with cb2:
    if r9["Ds"] >= gain_min:
        st.success(
            f"Gain requirement met\n\n"
            f"Element spacing: {r9['d']:.4f} lam = {r9['d_mm']:.2f} mm\n\n"
            f"Element directivity: {r9['ed']:.2f} dBi\n\n"
            f"Scan loss at +/-{scan_main} deg: {r9['sl']:.3f} dB\n\n"
            f"Total elements required: **{r9['N']}**\n\n"
            f"Peak directivity: {r9['Dp']:.2f} dBi\n\n"
            f"At +/-{scan_main} deg scan: {r9['Ds']:.2f} dBi >= {gain_min} dBi"
        )
    else:
        st.error(f"Directivity at scan edge ({r9['Ds']:.2f} dBi) is below {gain_min} dBi!")
# ─── PART c ──────────────────────────────────────────
st.markdown("---")
st.header("Part c - Array Layout & Radiation Patterns")
st.subheader(f"Hexagonal Array Layout - +/-{scan_main} deg Scan")
lc1, lc2 = st.columns([1, 1])
with lc1:
    fig1 = plot_layout(r9["N"], r9["d_mm"],
                       f"+/-{scan_main} deg | N = {r9['N']} | d = {r9['d_mm']:.2f} mm")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
with lc2:
    st.markdown(
        f"| Parameter | Value |\n|---|---|\n"
        f"| Grid Type | Hexagonal |\n"
        f"| Aperture Shape | Circular |\n"
        f"| Total Elements | **{r9['N']}** |\n"
        f"| Element Spacing | **{r9['d_mm']:.2f} mm** ({r9['d']:.4f} lam) |\n"
        f"| Peak Directivity | **{r9['Dp']:.2f} dBi** |\n"
        f"| Dir. at +/-{scan_main} deg | **{r9['Ds']:.2f} dBi** |\n"
        f"| GL at Boresight | {r9['glb']:.2f} deg |\n"
        f"| GL at Scan Edge | {r9['gls']:.2f} deg |"
    )
st.subheader(f"Radiation Patterns - +/-{scan_main} deg Scan")
fig2 = plot_patterns(r9, scan_main, f"f = {freq_ghz} GHz | N = {r9['N']}", gain_min)
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)
st.subheader("Supporting Analysis")
sc1, sc2 = st.columns(2)
with sc1:
    fig3 = plot_n_vs_dir(r9, scan_main, f"+/-{scan_main} deg", gain_min, elem_eff)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)
with sc2:
    fig4 = plot_dir_gl(r9, scan_main, f"+/-{scan_main} deg", elem_eff, horn_k)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)
fig5 = plot_bandwidth(r9, freq_ghz, freq_bw)
st.pyplot(fig5, use_container_width=True)
plt.close(fig5)
# ─── PART d ──────────────────────────────────────────
st.markdown("---")
st.header(f"Part d - Complexity Reduction: +/-{scan_main} deg to +/-{scan_red} deg")
pd1, pd2 = st.columns([1.3, 1])
with pd1:
    df_d = pd.DataFrame({
        "Parameter": [
            "Element Spacing (d/lam)",
            "Element Spacing (mm)",
            "Element Directivity (dBi)",
            "Scan Loss (dB)",
            "Number of Elements",
            "Peak Directivity (dBi)",
            "Dir. at Scan Edge (dBi)",
            "Grating-Lobe at Boresight (deg)",
        ],
        f"+/-{scan_main} deg": [
            f"{r9['d']:.4f}",
            f"{r9['d_mm']:.3f}",
            f"{r9['ed']:.2f}",
            f"{r9['sl']:.3f}",
            str(r9["N"]),
            f"{r9['Dp']:.2f}",
            f"{r9['Ds']:.2f}",
            f"{r9['glb']:.2f}",
        ],
        f"+/-{scan_red} deg": [
            f"{r6['d']:.4f}",
            f"{r6['d_mm']:.3f}",
            f"{r6['ed']:.2f}",
            f"{r6['sl']:.3f}",
            str(r6["N"]),
            f"{r6['Dp']:.2f}",
            f"{r6['Ds']:.2f}",
            f"{r6['glb']:.2f}",
        ],
        "Change": [
            f"+{spacing_inc_pct:.1f}%",
            f"+{r6['d_mm']-r9['d_mm']:.2f} mm",
            f"+{r6['ed']-r9['ed']:.2f} dB",
            f"{r6['sl']-r9['sl']:+.3f} dB",
            f"-{r9['N']-r6['N']} ({reduction_pct:.1f}%)",
            f"{r6['Dp']-r9['Dp']:+.2f} dBi",
            f"{r6['Ds']-r9['Ds']:+.2f} dBi",
            f"+{r6['glb']-r9['glb']:.2f} deg",
        ]
    })
    st.dataframe(df_d, use_container_width=True, hide_index=True)
with pd2:
    st.warning(
        f"Reducing scan from +/-{scan_main} deg to +/-{scan_red} deg:\n\n"
        f"Spacing increases by {spacing_inc_pct:.1f}% "
        f"({r9['d_mm']:.2f} mm to {r6['d_mm']:.2f} mm)\n\n"
        f"Element count: {r9['N']} to {r6['N']} "
        f"= {reduction_pct:.0f}% fewer T/R modules\n\n"
        f"Scan loss reduced by {r9['sl']-r6['sl']:.3f} dB\n\n"
        f"Gain still met at scan edge: {r6['Ds']:.2f} dBi >= {gain_min} dBi\n\n"
        f"Fewer elements = lower cost, weight, DC power, and manufacturing complexity."
    )
fig6 = plot_comparison(r9, r6, scan_main, scan_red)
st.pyplot(fig6, use_container_width=True)
plt.close(fig6)
st.subheader("Array Layout Comparison")
dl1, dl2 = st.columns(2)
with dl1:
    fig7 = plot_layout(r9["N"], r9["d_mm"],
                       f"+/-{scan_main} deg | N = {r9['N']} | d = {r9['d_mm']:.2f} mm")
    st.pyplot(fig7, use_container_width=True)
    plt.close(fig7)
with dl2:
    fig8 = plot_layout(r6["N"], r6["d_mm"],
                       f"+/-{scan_red} deg | N = {r6['N']} | d = {r6['d_mm']:.2f} mm")
    st.pyplot(fig8, use_container_width=True)
    plt.close(fig8)
st.subheader(f"Radiation Patterns - +/-{scan_red} deg Scan")
fig9 = plot_patterns(r6, scan_red, f"f = {freq_ghz} GHz | N = {r6['N']}", gain_min)
st.pyplot(fig9, use_container_width=True)
plt.close(fig9)
st.markdown("---")
st.success(
    f"Design complete. "
    f"Peak directivity: {r9['Dp']:.2f} dBi | "
    f"Directivity at +/-{scan_main} deg: {r9['Ds']:.2f} dBi >= {gain_min} dBi required."
)

