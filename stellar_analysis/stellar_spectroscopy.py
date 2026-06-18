import numpy as np
import matplotlib.pyplot as plt

# 1. Global Physical Constants (SI Units)
H = 6.62607015e-34  # Planck's constant (J.s)
C = 299792458       # Speed of light (m/s)
KB = 1.380649e-23   # Boltzmann's constant (J/K)
WIEN_CONSTANT = 2.897771955e-3  # Wien's displacement constant (m.K)

def planck_formula(wavelength, temperature):
    """
    Calculates Spectral Radiance using Planck's Law.
    
    Parameters:
    wavelength (numpy.ndarray): Wavelengths in meters.
    temperature (float): Effective stellar temperature in Kelvin.
    
    Returns:
    numpy.ndarray: Spectral radiance (W * sr^-1 * m^-3).
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0 Kelvin.")
        
    numerator = 2 * H * (C**2)
    exponent = (H * C) / (wavelength * KB * temperature)
    
    # Preventing overflow errors in exp
    exponent = np.clip(exponent, None, 700) 
    
    intensity = numerator / (wavelength**5 * (np.exp(exponent) - 1))
    return intensity

def get_wien_peak(temperature):
    """Calculates peak wavelength using Wien's Displacement Law."""
    return WIEN_CONSTANT / temperature

# --- Step 1: Modeling Real Stars Blackbody Radiation ---
# Target Stars: Rigel (B-type Blue Supergiant), The Sun (G-type), Betelgeuse (M-type Red Supergiant)
stars = {
    "Rigel (Blue Supergiant)": {"T": 12100, "color": "#1f77b4"},
    "The Sun (G-type Star)": {"T": 5778, "color": "#ff7f0e"},
    "Betelgeuse (Red Supergiant)": {"T": 3500, "color": "#d62728"}
}

wavelengths = np.linspace(50e-9, 2000e-9, 1000) # 50nm to 2000nm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Blackbody Radiation Spectra
for name, properties in stars.items():
    T = properties["T"]
    color = properties["color"]
    
    # Calculate Planck Spectrum
    intensity = planck_formula(wavelengths, T)
    ax1.plot(wavelengths * 1e9, intensity, label=f'{name} ({T} K)', color=color, linewidth=2.5)
    
    # Calculate and plot Wien's Peak
    lambda_max = get_wien_peak(T)
    peak_intensity = planck_formula(np.array([lambda_max]), T)[0]
    ax1.vlines(lambda_max * 1e9, 0, peak_intensity, colors=color, linestyles='dashed', alpha=0.5)
    ax1.scatter(lambda_max * 1e9, peak_intensity, color=color, s=40, zorder=5)

ax1.set_title("Stellar Blackbody Radiation Spectra & Wien's Law Peaks", fontsize=12, fontweight='bold')
ax1.set_xlabel("Wavelength (nm)", fontsize=11)
ax1.set_ylabel("Spectral Radiance ($W \cdot sr^{-1} \cdot m^{-3}$)", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- Step 2: Simulating an Eclipsing Binary Light Curve ---
# Simulating a simple binary system where a cooler star eclipses a hotter star
phases = np.linspace(0, 2, 500) # 2 complete orbital periods
base_flux = 1.0

# Simplified model: Primary eclipse (deeper), Secondary eclipse (shallower)
flux = np.ones_like(phases)
# Primary Eclipse at phase 0.5 and 1.5
primary_eclipse = (np.abs(phases - 0.5) < 0.1) | (np.abs(phases - 1.5) < 0.1)
flux[primary_eclipse] -= 0.3 * np.cos((phases[primary_eclipse] % 1.0 - 0.5) * np.pi / 0.2)**2

# Secondary Eclipse at phase 1.0 and 2.0 (0.0)
secondary_eclipse = (np.abs(phases - 1.0) < 0.08) | (phases < 0.08) | (phases > 1.92)
mask = (np.abs(phases - 1.0) < 0.08)
flux[mask] -= 0.1 * np.cos((phases[mask] - 1.0) * np.pi / 0.16)**2
mask2 = (phases < 0.08)
flux[mask2] -= 0.1 * np.cos((phases[mask2]) * np.pi / 0.16)**2
mask3 = (phases > 1.92)
flux[mask3] -= 0.1 * np.cos((phases[mask3] - 2.0) * np.pi / 0.16)**2

# Plot 2: Eclipsing Binary Light Curve
ax2.plot(phases, flux, color='#2ca02c', linewidth=2.5, label='Simulated Data')
ax2.set_title("Simulated Eclipsing Binary Light Curve", fontsize=12, fontweight='bold')
ax2.set_xlabel("Orbital Phase", fontsize=11)
ax2.set_ylabel("Normalized Flux", fontsize=11)
ax2.set_ylim(0.6, 1.1)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Add some visual annotations for astrophysics context
ax2.annotate('Primary Eclipse\n(Hot Star Occluded)', xy=(0.5, 0.7), xytext=(0.1, 0.65),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
ax2.annotate('Secondary Eclipse\n(Cool Star Occluded)', xy=(1.0, 0.9), xytext=(1.2, 0.8),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig("stellar_analysis_portfolio.png", dpi=300)
plt.show()