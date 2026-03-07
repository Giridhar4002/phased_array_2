---
title: Phased Array Antenna Design
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
---

# Phased Array Antenna Design Tool
Computes required array peak directivity, element spacing, element gain, number of elements, and plots the array layout and radiation patterns based on CICAD 2025 specs.

# Phased Array Antenna Design Application

This Streamlit application calculates and visualizes the design parameters for a phased array antenna based on specific user requirements, including an analysis of complexity reduction based on scan angles.

## Default Specifications Addressed:
* **Frequency:** 44.5 GHz +/- 1.0 GHz
* **Grid:** Hexagonal
* **Aperture Shape:** Circular
* **Feed Element:** Potter horn (70% aperture efficiency)
* **Scan Region:** +/- 9 degrees (with comparison to +/- 6 degrees)
* **Minimum Gain over coverage:** 40 dBi
* **Amplitude Taper:** 10 dB across the array aperture

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`