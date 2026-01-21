
"""
How did we choose this parameters and approximated this model?

L = 0.267 m, the distances can be approximated as lf = lr = 0.1335 m.

Firstly, we need to estimate the vertical load Fz on each tire using static weight distribution:
Fzf = (lr/L) m g, Fzr =(lf/L) m g. The mass taken from the gazebo model m= 1.415kg, μ = 0.9
The peak lateral force D was estimated as D = μFz. 

The shape factor C was selected based on literature values for small-scale rubber tires, 1.2-1.4

The slip angle peaks between 5◦ and 8◦ before the force begins to decline due to
tire saturation, α = 6.21◦. 

Finally, the stiffness factor B was calculated from
the small-angle linearized cornering stiffness relationship Cα = BCD, where Cα
represents the slope of the linear tire model near α = 0. The cornering stiffness
Cα was approximated from empirical ratios found in scaled-vehicle studies. 
Note that the values chosen are the same for both front and rear tires.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
B = 0.425       # stiffness factor
C = 1.3             # shape factor
D = 6.246           # peak factor 
E = -0.5            # curvature factor, not sure if this is correct

# Slip angles (deg → rad)
alpha_deg = np.linspace(0,15, 500)
alpha = np.radians(alpha_deg)

# Models
Fy_pacejka = D * np.sin(C * np.arctan(B * alpha_deg))
Fy_pacejka_E = D * np.sin(C * np.arctan(B * alpha_deg - E * (B * alpha_deg - np.arctan(B * alpha_deg))))
C_alpha = D * C * B
Fy_linear = C_alpha * alpha_deg

# Peak of standard Pacejka
idx_peak = np.argmax(Fy_pacejka)
alpha_peak = alpha_deg[idx_peak]
Fy_peak = Fy_pacejka[idx_peak]


# Peak of standard Pacejka
idx_peakE = np.argmax(Fy_pacejka_E)
alpha_peakE = alpha_deg[idx_peakE]
Fy_peakE = Fy_pacejka_E[idx_peakE]

# Plot
plt.figure(figsize=(9,6))
plt.plot(alpha_deg, Fy_pacejka, label="Simplified Pacejka (no E)", linewidth=2)
plt.plot(alpha_deg, Fy_pacejka_E, label="Full Pacejka (with E)", linewidth=2)
plt.plot(alpha_deg, Fy_linear, '--', label="Linear Tire Model", linewidth=2)

# Add peak dotted lines
plt.axvline(alpha_peak, color='k', linestyle=':', alpha=0.7)
plt.axhline(Fy_peak, color='k', linestyle=':', alpha=0.7)
plt.scatter(alpha_peak, Fy_peak, color='blue', zorder=3)
#plt.text(alpha_peak+0.3, Fy_peak-0.3, f"Peak @ α={alpha_peak:.2f}°, Fy={Fy_peak:.2f} N")


# Add peak dotted lines
plt.axvline(alpha_peakE, color='k', linestyle=':', alpha=0.7)
plt.axhline(Fy_peakE, color='k', linestyle=':', alpha=0.7)
plt.scatter(alpha_peakE, Fy_peakE, color='orange', zorder=3)
#plt.text(alpha_peak+0.3, Fy_peak-0.3, f"Peak @ α={alpha_peak:.2f}°, Fy={Fy_peak:.2f} N")



# 1. Draw an angle arc representing C_alpha (slope line)
origin = [0, 0]
radius = 2.0  # arc radius for visibility
angle = np.arctan(C_alpha)  # small-angle slope in radians
arc_angles = np.linspace(0, angle, 50)
arc_x = radius * np.cos(arc_angles)
arc_y = radius * np.sin(arc_angles)
plt.plot(arc_x, arc_y, 'g', lw=1.5)
plt.text(radius * np.cos(angle/2) + 0.3,
         radius * np.sin(angle/2),
         r"$C_\alpha = DCB$",
         fontsize=12, color='g')

# 2. Add D value below peak point
a= (alpha_peak+alpha_peakE)/2
plt.text(a , Fy_peak - 0.6,
         r"$D = \mu F_z$".format(D),
         fontsize=11, color='black',
         ha='center')



# Styling
#plt.title("Lateral Force vs Slip Angle (Pacejka vs Linear Tire Model)")
plt.xlabel("Slip Angle α [deg]")
plt.ylabel("Vertical Load Fz [N]")
plt.ylim((-0.0,7.7))
plt.xlim((-0.0,15))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
