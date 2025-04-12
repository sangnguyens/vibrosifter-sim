# --- Placeholder Constants for Models ---
EMPIRICAL_CONSTANT_C = 0.150  # Speed constant
GRAVITY_G = 9.81  # m/s^2
# --- Blocking Model Constants ---
BLOCKING_FINF_MIN = 0.1  # Minimum possible f_inf
BLOCKING_FINF_MAX = 0.8  # Maximum possible f_inf
BLOCKING_K_SENSITIVITY = 0.5  # How quickly f_inf increases with K
BLOCKING_K_MIDPOINT = 3.0  # Toss indicator K for midpoint f_inf

F_max = 15000.0  # Default value
top_ecc_m = 0.05  # Default value
bottom_ecc_m = 0.1  # Default value
lead_angle = 30.0  # Default value
min_psd = 0.07  # Default value
mean_psd = 0.1  # Default value
max_psd = 0.2  # Default value
bulk_density = 1.05  # Default value
stiffness_p = 2000.0  # Default value
num_springs = 6  # Default value
freq_current = 50.0  # Default value

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# SciPy no longer needed

# --- Page Configuration ---
st.set_page_config(
    page_title="Vibrosifter Sim (Blocking = Efficiency)",
    page_icon="⚙️📈",
    layout="wide",
)


# --- Helper Functions ---
def calculate_natural_frequency(k_total, m_total):
    if m_total <= 0 or k_total <= 0:
        return 0.0
    omega_n = np.sqrt(k_total / m_total)
    f_n = omega_n / (2 * np.pi)
    return f_n


def calculate_component_amplitude(F_comp, k_total, m_total, f_op, f_n):
    if k_total <= 0 or f_n <= 1e-6 or m_total <= 0:
        return 0.0
    if abs(f_op - f_n) < 1e-6:
        return 1.0  # Resonance indicator
    static_def = F_comp / k_total
    freq_ratio = f_op / f_n
    denom = abs(1 - freq_ratio**2)
    if denom < 1e-6:
        return 1.0  # Resonance indicator
    dyn_amp_factor = 1 / denom
    amp_m = static_def * dyn_amp_factor
    return max(0.0, amp_m)


def calculate_forces(top_moment, bot_moment, lead_angle_deg, freq_hz):
    if freq_hz <= 0:
        return 0.0, 0.0, 0.0
    omega_sq = (2 * np.pi * freq_hz) ** 2
    F_top = top_moment * omega_sq
    F_bottom = bot_moment * omega_sq
    delta_rad = np.radians(lead_angle_deg)
    F_total_sq = F_top**2 + F_bottom**2 + 2 * F_top * F_bottom * np.cos(delta_rad)
    F_total = np.sqrt(max(0, F_total_sq))
    return F_top, F_bottom, F_total


def calculate_transport_speed_enhanced(A_h_mm, A_v_mm, freq_hz, lead_angle, C):
    A_eff_mm = np.sqrt(A_h_mm**2 + A_v_mm**2)
    if lead_angle <= 0 or C <= 0 or A_eff_mm <= 1e-6:
        return 0.0
    delta_deg = min(max(0, lead_angle), 90)
    delta_rad = np.radians(delta_deg)
    speed_mps = C * (A_eff_mm/1000 * (2 * np.pi * freq_hz)) * np.cos(delta_rad)
    return max(0.0, speed_mps)


def calculate_throughput_kg_h_base(
    speed_mps, diameter_m, bed_depth_m, density_kg_m3, hole_fraction_area
):
    """Calculates BASE mass throughput assuming 100% open area"""
    if speed_mps <= 0 or diameter_m <= 0 or bed_depth_m <= 0:
        return 0.0
    vol_flow = speed_mps * diameter_m * bed_depth_m
    mass_flow = vol_flow * density_kg_m3 * 3600 * hole_fraction_area
    return mass_flow


def calculate_steady_state_blocking_f(
    toss_indicator_K, f_inf_min, f_inf_max, K_sensitivity, K_midpoint
):
    """Calculates f_inf based on Toss Indicator K using a sigmoid function."""
    if toss_indicator_K < 0:
        toss_indicator_K = 0
    exponent_term = -K_sensitivity * (toss_indicator_K - K_midpoint)
    if exponent_term > 700:
        sigmoid_denominator = 1.0
    elif exponent_term < -700:
        sigmoid_denominator = float("inf")
    else:
        sigmoid_denominator = 1.0 + math.exp(exponent_term)
    if sigmoid_denominator == float("inf"):
        f_inf = f_inf_max
    else:
        f_inf = f_inf_min + (f_inf_max - f_inf_min) / sigmoid_denominator
    return max(f_inf_min, min(f_inf, f_inf_max))


def recommend_mesh_size(min_particle_mm):
    """Recommends mesh size based on min particle size for scalping. Returns value_mm (aperture estimate)."""
    if min_particle_mm > 0.1:
        safety_factor = 0.8
    else:
        safety_factor = 0.7
    mesh_mm = min_particle_mm * safety_factor
    practical_min_mesh_mm = 0.010
    recommended_mesh_aperture_mm = max(practical_min_mesh_mm, mesh_mm)
    return recommended_mesh_aperture_mm


# --- Simulation Function ---
def simulate_particle_path(
    diameter, lead_angle_deg, amplitude_eff_mm, step_scale_factor, max_steps=50000
):
    """Simulates a conceptual particle path based on lead angle. Returns lists of x and y coordinates."""
    radius = diameter / 2.0
    lead_angle_rad = np.radians(lead_angle_deg)
    amplitude_eff_m = amplitude_eff_mm / 1000.0

    r_current = 0.01 * radius
    theta_current = 0.0
    x_path = [r_current * np.cos(theta_current)]
    y_path = [r_current * np.sin(theta_current)]
    steps = 0
    path_warning = None

    while r_current < radius and steps < max_steps:
        base_step = step_scale_factor * amplitude_eff_m
        delta_r = base_step * np.cos(lead_angle_rad)
        delta_arc = base_step * np.sin(lead_angle_rad)
        if r_current < 1e-6:
            delta_theta = 0
        else:
            delta_theta = delta_arc / r_current
        r_current = max(0.001 * radius, r_current + delta_r)
        theta_current += delta_theta
        x_path.append(r_current * np.cos(theta_current))
        y_path.append(r_current * np.sin(theta_current))
        steps += 1

    if r_current < radius and steps == max_steps:
        path_warning = (
            f"Max steps ({max_steps}) reached. Path incomplete (high lead angle?)."
        )
    elif steps < max_steps and len(x_path) > 1:  # Normalize last step
        try:  # Add try-except for potential math errors in normalization
            last_x, last_y = x_path[-1], y_path[-1]
            prev_x, prev_y = x_path[-2], y_path[-2]
            dx, dy = last_x - prev_x, last_y - prev_y
            dr_sq = dx * dx + dy * dy
            D_intersect = prev_x * dy - prev_y * dx
            discriminant = (radius**2) * dr_sq - D_intersect**2
            if discriminant >= 0 and dr_sq > 1e-9:
                sgn_dy = 1 if dy >= 0 else -1
                sqrt_discriminant = np.sqrt(discriminant)
                x1 = (D_intersect * dy + sgn_dy * dx * sqrt_discriminant) / dr_sq
                y1 = (-D_intersect * dx + abs(dy) * sqrt_discriminant) / dr_sq
                x2 = (D_intersect * dy - sgn_dy * dx * sqrt_discriminant) / dr_sq
                y2 = (-D_intersect * dx - abs(dy) * sqrt_discriminant) / dr_sq
                dist1_sq = (x1 - last_x) ** 2 + (y1 - last_y) ** 2
                dist2_sq = (x2 - last_x) ** 2 + (y2 - last_y) ** 2
                if dist1_sq < dist2_sq:
                    x_path[-1], y_path[-1] = x1, y1
                else:
                    x_path[-1], y_path[-1] = x2, y2
            else:  # Fallback if normalization fails
                x_path[-1] = radius * np.cos(theta_current)
                y_path[-1] = radius * np.sin(theta_current)
        except Exception:  # Catch any math errors during normalization
            x_path[-1] = radius * np.cos(theta_current)
            y_path[-1] = radius * np.sin(theta_current)

    return x_path, y_path, path_warning


# --- Streamlit App Layout ---
st.title("🧱📉 Vibrosifter Simulation")
# st.markdown("Estimates performance where **steady-state blocking (`f_∞`) directly represents efficiency/capacity**. Calculates dynamics & blocking based on vibration intensity (`K`).")
# st.warning(f"""**Disclaimer:** Amplitude calc simplified & force-limited. Speed uses estimated `C`. **Blocking model (`f∞=fn(K)`) uses placeholder constants** (`f∞_min`, `f∞_max`, `K_sens`, `K_mid`) and dictates efficiency. Validation essential.""")
st.markdown("---")

# --- Input Parameters ---
st.sidebar.header("Input Parameters")

st.sidebar.subheader("Model Tuning Parameters")
empirical_constant_c_horizontal = st.sidebar.number_input(
    "Speed Constant (C)",
    min_value=0.001,
    max_value=0.3,
    value=EMPIRICAL_CONSTANT_C,
    step=0.01,
    format="%.3f",
    help="For `v ≈ C*A_eff*omega*cos(δ)`",
)
st.sidebar.markdown("**Steady-State Blocking Model (`f∞ = fn(K)`):**")
hole_fraction_area = st.sidebar.slider(
    r"Opening Frac. of Screen, $\alpha_{op}$",
    min_value=0.001,
    max_value=1.0,
    value=0.26,
    step=0.01,
    help="Design Opening fraction of screen, 0.26 for 500 Mesh",
)
blocking_finf_min = st.sidebar.slider(
    "Min Equil. Free Fraction (f∞_min)",
    min_value=0.01,
    max_value=0.2,
    value=BLOCKING_FINF_MIN,
    step=0.01,
    help="Minimum possible f∞ (at K=0).",
)
blocking_finf_max = st.sidebar.slider(
    "Max Equil. Free Hole Fraction",
    min_value=0.05,
    max_value=1.0,
    value=BLOCKING_FINF_MAX,
    step=0.01,
    help="Maximum possible free hole available.",
)
blocking_k_sensitivity = st.sidebar.number_input(
    "Blocking K Sensitivity",
    min_value=0.05,
    max_value=2.0,
    value=BLOCKING_K_SENSITIVITY,
    step=0.05,
    help="How quickly f∞ increases with Toss Indicator K.",
)
blocking_k_midpoint = st.sidebar.number_input(
    "Blocking K Midpoint",
    min_value=0.5,
    max_value=10.0,
    value=BLOCKING_K_MIDPOINT,
    step=0.1,
    help="Toss Indicator K where f∞ is halfway between min/max.",
)
st.sidebar.caption("**Note Tuning Params:** Require experimental calibration.")

motor_max_exciting_force_n = st.sidebar.number_input(
    "Motor Max Exciting Force (F_max) (N)",
    min_value=500.0,
    max_value=20000.0,
    value=F_max,
    step=100.0,
    help="Maximum centrifugal force rating of the motor.",
)

st.sidebar.subheader("Eccentric Weight Settings")
top_mass_moment_kgm = st.sidebar.number_input(
    "Top Mass Moment (m*r) (kg·m)",
    min_value=0.0,
    max_value=1.50,
    value=top_ecc_m,
    step=0.05,
    format="%.3f",
    help="Influences Horizontal Amplitude.",
)
bottom_mass_moment_kgm = st.sidebar.number_input(
    "Bottom Mass Moment (m*r) (kg·m)",
    min_value=0.0,
    max_value=1.50,
    value=bottom_ecc_m,
    step=0.05,
    format="%.3f",
    help="Influences Vertical Amplitude.",
)
lead_angle_physical_deg = st.sidebar.slider(
    "Physical Lead Angle (δ) (degrees)",
    min_value=0.0,
    max_value=90.0,
    value=lead_angle,
    step=5.0,
    help="PHYSICAL angle set between weights.",
)

st.sidebar.subheader("Feed Properties")
# If min size is truly needed only for aperture suggestion, add it back:
min_granule_mm_for_suggestion = st.sidebar.number_input(
    "Min Granule Size (for suggestion) (mm)",
    min_value=0.0010,
    max_value=0.10,
    value=min_psd,
    step=0.01,
    format="%.3f",
    help="ONLY used to suggest default screen aperture.",
)
# mean_granule_mm = st.sidebar.number_input(
#     "Feed Particle Diameter (mm)", min_value=0.010, max_value=50.0, value=mean_psd, step=0.1, format="%.3f",
#     help="Mainly for context, not used in calcs now."
# )
bulk_density_gcm3 = st.sidebar.number_input(
    "Feed Bulk Density (g/cm³ or t/m³)",
    min_value=0.5,
    max_value=1.30,
    value=bulk_density,
    step=0.05,
)

st.sidebar.subheader("Machine & System")
screen_diameter_m = st.sidebar.number_input(
    "Screen Diameter (m)", min_value=0.2, max_value=3.0, value=1.0, step=0.1
)
suggested_aperture_mm = recommend_mesh_size(
    min_granule_mm_for_suggestion
)  # Use the suggestion input
# aperture_size_mm_input = st.sidebar.number_input( # User input aperture in mm
#     "Screen Aperture Size (a) (mm)", min_value=0.020, max_value=50.0, value=suggested_aperture_mm, step=0.01, format="%.3f",
#     help="Actual opening size (informational only)."
# )
machine_vibrating_mass_kg = st.sidebar.number_input(
    "Machine Vibrating Mass (Dry) (kg)",
    min_value=40.0,
    max_value=1500.0,
    value=80.0,
    step=10.0,
    help="Excludes material load.",
)
stiffness_per_spring_N_m = st.sidebar.number_input(
    "Stiffness per Spring (N/m)",
    min_value=1000.0,
    max_value=5000.0,
    value=stiffness_p,
    step=1000.0,
)
number_of_springs = st.sidebar.number_input(  # Corrected typo previously
    "Number of Support Springs",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
    help="Total number of springs.",
)
est_bed_depth_mm = st.sidebar.number_input(
    "Estimated Material Bed Depth (mm)",
    min_value=1.0,
    max_value=50.0,
    value=15.0,
    step=1.0,
    help="Influences total mass.",
)

st.sidebar.subheader("Operational Parameters")
frequency_hz = st.sidebar.slider(
    "Operating Frequency (Hz)",
    min_value=10.0,
    max_value=90.0,
    value=freq_current,
    step=1.0,
)
# --- Calculations ---
# Unit conversions
suggested_aperture_mm = recommend_mesh_size(
    min_granule_mm_for_suggestion
)  # Use the suggestion input

bulk_density_kg_m3 = bulk_density_gcm3 * 1000.0
bed_depth_m = est_bed_depth_mm / 1000.0
screen_radius_m = screen_diameter_m / 2.0
screen_area_m2 = math.pi * (screen_radius_m**2)

# Mass Calcs
material_volume_m3 = screen_area_m2 * bed_depth_m
material_mass_kg = material_volume_m3 * bulk_density_kg_m3
total_vibrating_mass_kg = machine_vibrating_mass_kg + material_mass_kg

# System Props
k_total_N_m = stiffness_per_spring_N_m * number_of_springs
f_n_hz = calculate_natural_frequency(k_total_N_m, total_vibrating_mass_kg)
frequency_ratio = frequency_hz / f_n_hz if f_n_hz > 1e-6 else float("inf")

# Forces
F_top_theo_n, F_bottom_theo_n, F_total_theo_n = calculate_forces(
    top_mass_moment_kgm, bottom_mass_moment_kgm, lead_angle_physical_deg, frequency_hz
)
# Force Limiting Check
is_force_limited = False
F_top_actual_n = F_top_theo_n
F_bottom_actual_n = F_bottom_theo_n
F_total_actual_n = F_total_theo_n
if F_total_theo_n > motor_max_exciting_force_n and F_total_theo_n > 1e-6:
    is_force_limited = True
    scaling_factor = motor_max_exciting_force_n / F_total_theo_n
    F_top_actual_n = F_top_theo_n * scaling_factor
    F_bottom_actual_n = F_bottom_theo_n * scaling_factor
    F_total_actual_n = motor_max_exciting_force_n
F_total_display_kn = F_total_actual_n / 1000.0
motor_max_force_kn = motor_max_exciting_force_n / 1000.0

# Amplitudes
A_h_m = calculate_component_amplitude(
    F_top_actual_n, k_total_N_m*0.2, total_vibrating_mass_kg, frequency_hz, f_n_hz
)
A_v_m = calculate_component_amplitude(
    F_bottom_actual_n, k_total_N_m, total_vibrating_mass_kg, frequency_hz, f_n_hz
)

is_resonant_h = A_h_m >= 0.999
is_resonant_v = A_v_m >= 0.999
is_resonant = is_resonant_h or is_resonant_v
if is_resonant_h:
    A_h_mm = float("inf")
else:
    A_h_mm = A_h_m * 1000.0
if is_resonant_v:
    A_v_mm = float("inf")
else:
    A_v_mm = A_v_m * 1000.0

# Calculate effective amplitude for speed calc (avoid error if resonant)
A_eff_mm_calc = 0.0 if is_resonant else np.sqrt(A_h_mm**2 + A_v_mm**2)

# calculate hole area
# a_hole = np.pi * (suggested_aperture_mm / 2 * 10**(-6))
# a_screen = np.pi * (screen_diameter_m ** 2 / 4)

# Calculate Speed & Base Throughput
if is_resonant:
    calculated_speed_mps = float("inf")
    throughput_kg_h_base = 0.0
    throughput_t_h_base = 0.0
else:
    calculated_speed_mps = calculate_transport_speed_enhanced(
        A_h_mm,
        A_v_mm,
        frequency_hz,
        lead_angle_physical_deg,
        empirical_constant_c_horizontal,
    )
    throughput_kg_h_base = calculate_throughput_kg_h_base(
        calculated_speed_mps,
        screen_diameter_m,
        bed_depth_m,
        bulk_density_kg_m3,
        hole_fraction_area,
    )
    throughput_t_h_base = throughput_kg_h_base / 1000.0

# Calculate Toss Indicator K & Steady-State Blocking f_inf
if is_resonant_v or frequency_hz <= 1e-6:
    toss_indicator_K = 0.0
else:
    toss_indicator_K = (4 * (math.pi**2) * (frequency_hz**2) * A_v_m) / GRAVITY_G
steady_state_f_inf = calculate_steady_state_blocking_f(
    toss_indicator_K,
    blocking_finf_min,
    blocking_finf_max * hole_fraction_area,
    blocking_k_sensitivity,
    blocking_k_midpoint,
)

# Calculate ACTUAL Throughput adjusted for blocking
throughput_kg_h_actual = throughput_kg_h_base * steady_state_f_inf / hole_fraction_area
throughput_t_h_actual = throughput_kg_h_actual / 1000.0

# Efficiency is now directly represented by f_inf
screen_efficiency_perc = steady_state_f_inf * 100.0

# Format Aperture Size for Display
if suggested_aperture_mm < 0.1:
    aperture_display_val = suggested_aperture_mm * 1000.0
    aperture_display_unit = "µm"
    aperture_display_format = ".0f"
else:
    aperture_display_val = suggested_aperture_mm
    aperture_display_unit = "mm"
    aperture_display_format = ".3f"
aperture_display_str = (
    f"{aperture_display_val:{aperture_display_format}} {aperture_display_unit}"
)

# --- Display Results & Recommendations ---
# st.warning(f"""**Disclaimer:** Amplitude calc simplified & force-limited. Speed uses estimated `C`. **Efficiency IS the calculated steady-state free fraction (`f∞`) based on a blocking model with placeholder constants.** Validation essential.""")

st.header(":green[System Dynamics & Calculated Vibration]")
col_dyn1, col_dyn2, col_dyn3 = st.columns(3)
with col_dyn1:
    st.metric(
        label="Total Vib. Mass ($M_{total}$)", value=f"{total_vibrating_mass_kg:.1f} kg"
    )
    st.caption(
        f"Machine ({machine_vibrating_mass_kg}kg) + Material ({material_mass_kg:.1f}kg)"
    )
with col_dyn2:
    st.metric(label="Natural Freq ($f_n$)", value=f"{f_n_hz:.2f} Hz")
    st.caption(r"$f_n \approx \frac{1}{2\pi} \sqrt{k/M}$")
with col_dyn3:
    st.metric(label="Freq Ratio ($f/f_n$)", value=f"{frequency_ratio:.2f}")
    # ... resonance warning/info
col_vib1, col_vib2, col_vib3 = st.columns(3)
with col_vib1:
    st.metric(
        label="Est. $A_h$ / $A_v$",
        value=f"{A_h_mm:.2f}/{A_v_mm:.2f} mm" if not is_resonant else "Resonant!",
    )
    st.caption(r"$A \approx \frac{F_{act}/k}{|1-(f/f_n)^2|}$")
with col_vib2:
    st.metric(
        label="Toss Indicator ($K$)",
        value=f"{toss_indicator_K:.2f}" if not is_resonant_v else "N/A",
    )
    st.caption(r"$K \approx 4\pi^2 f^2 A_v/g$")
with col_vib3:
    st.metric(label="$F_{actual}$", value=f"{F_total_display_kn:.2f} kN")
    force_percentage = (
        (F_total_actual_n / motor_max_exciting_force_n) * 100
        if motor_max_exciting_force_n > 0
        else 0
    )
    st.caption(
        f"$F_{{act}}$ / $F_{{max}}$ ({motor_max_force_kn:.1f} kN), ~{force_percentage:.0f}%"
    )
if is_force_limited:
    st.warning("Motor force limited!")
elif F_total_actual_n <= 1e-6:
    st.info("Enter weights/frequency.")
else:
    st.success(f"Force Feasible")

# --- NEW: Flow Pattern Visualizer Section ---
if plt:  # Only show if Matplotlib was imported successfully
    with st.expander(":blue[Visualize Conceptual Flow Pattern]", expanded=True):
        st.markdown("How lead angle affects the particle path (illustrative only).")
        vis_step_scale = st.slider(
            "Visualization Step Scale Factor (k_step)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            key="visualizer_step_scale",  # Unique key for this slider
            help="Tuning parameter: Adjusts step size in visualization. Does not affect calculations.",
        )

        # Use calculated effective amplitude if not resonant
        vis_amp_eff = (
            A_eff_mm_calc if not is_resonant else 1.0
        )  # Use a small default if resonant

        vis_x, vis_y, vis_warning = simulate_particle_path(
            screen_diameter_m,
            lead_angle_physical_deg,  # Use the main input lead angle
            vis_amp_eff,
            vis_step_scale,
        )

        if vis_warning:
            st.warning(vis_warning)

        fig_vis, ax_vis = plt.subplots(figsize=(3, 3))
        screen_circle_vis = plt.Circle(
            (0, 0),
            screen_diameter_m / 2.0,
            color="gray",
            fill=False,
            linestyle="--",
            linewidth=1,
        )
        ax_vis.add_patch(screen_circle_vis)
        ax_vis.plot(
            vis_x,
            vis_y,
            marker=".",
            markersize=1,
            linestyle="-",
            linewidth=0.8,
            label=f"Path (δ={lead_angle_physical_deg}°)",
        )
        if len(vis_x) > 0:
            ax_vis.plot(vis_x[0], vis_y[0], "go", markersize=4, label="Start")
            ax_vis.text(vis_x[0], vis_y[0], s="Start")
            ax_vis.plot(vis_x[-1], vis_y[-1], "ro", markersize=4, label="End")
            ax_vis.text(vis_x[-1], vis_y[-1], s="End")
        ax_vis.set_xlabel("X (m)")
        ax_vis.set_ylabel("Y (m)")
        ax_vis.set_title("Conceptual Particle Path")
        ax_vis.set_aspect("equal", adjustable="box")
        vis_limit = screen_diameter_m / 2.0 * 1.1
        ax_vis.set_xlim(-vis_limit, vis_limit)
        ax_vis.set_ylim(-vis_limit, vis_limit)
        ax_vis.grid(True, linestyle=":", alpha=0.5)
        # ax_vis.legend(loc='upper right')
        st.pyplot(fig_vis)
        st.caption(
            "Note: Simplified model ignoring friction, collisions, detailed dynamics. Illustrates directional trend vs. lead angle."
        )


st.header(":green[Estimated Performance]")
col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)  # Added 4th column back
with col_perf1:
    st.metric(
        label="Est. Speed ($v$)",
        value=f"{calculated_speed_mps:.3f} m/s" if not is_resonant else "Resonant!",
    )
    st.caption(r"$v \approx C A_{eff} f^2 \cos\delta\\A_{eff}=\sqrt{A_h^2 + A_v^2}$")
with col_perf2:
    st.metric(
        label="Est. Base Throughput ($Q_{base}$)",
        value=f"{throughput_kg_h_base:.0f} kg/h" if not is_resonant else "N/A",
    )
    st.caption(r"$Q_{base} \approx v D h \rho \alpha_{op}$")
with col_perf3:
    st.metric(
        label="Est. Actual Throughput ($Q_{actual}$)",
        value=f"{throughput_kg_h_actual:.0f} kg/h" if not is_resonant else "N/A",
    )
    st.caption(r"$Q_{actual} \approx Q_{base} \cdot f_{\infty}$")
with col_perf4:
    st.metric(
        label="Est. Efficiency ($f_∞$)",
        value=f"{screen_efficiency_perc/hole_fraction_area:.1f}%",
    )
    st.caption(f"Blocking Model ($K={toss_indicator_K:.2f}$)")


with st.expander(":blue[Show Blocking Model Formulas]"):
    st.markdown("**Steady-State Blocking Model (`f_∞ = fn(K)`):**")
    st.latex(r"K \approx \frac{4\pi^2 f^2 A_v}{g}")
    st.caption(f"Toss Indicator K calculated.")
    st.latex(
        r"f_{\infty} = f_{min} + \frac{f_{max}-f_{min}}{1+e^{-K_{sens}(K-K_{mid})}}"
    )
    st.caption(
        f"Calculated $f_{{∞}}$ using $K={toss_indicator_K:.2f}$ and tuning params."
    )


st.header(":green[Configuration Summary]")
col_setup1, col_setup2, col_setup3 = st.columns(3)
with col_setup1:
    st.subheader("Screen")
    st.metric(label="Diameter", value=f"{screen_diameter_m:.2f} m")
    st.metric(label="Aperture Size (a)", value=aperture_display_str)
with col_setup2:
    st.subheader("System")
    st.metric(label="Machine Mass", value=f"{machine_vibrating_mass_kg:.1f} kg")
    st.metric(label="Total Stiffness", value=f"{k_total_N_m/1000:.1f} kN/m")
with col_setup3:
    st.subheader("Material")
    st.metric(label="Bulk Density", value=f"{bulk_density_gcm3} g/cm³")
    st.metric(
        label="Particle Size (d)", value=f"{min_granule_mm_for_suggestion:.3f} mm"
    )


st.markdown("---")
