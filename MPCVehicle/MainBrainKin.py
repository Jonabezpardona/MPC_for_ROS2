'''
called Main Brain to be consistent with the main code for ROS, actually the Closed Loop is here
d_ocp, d_sim - stepsizes of the contoroller and simulation 

Control loop operates in time domain, while the controller operates in spatial domain.
When employing models that operate in two different domains, the spatial and
time domains, two main strategies can be considered for updating the control: 
time-based and distance-based.

The strategy consists of time-based updates, where the controller is executed 
at a constant frequency, i.e. every fixed time interval. This approach simplifies 
real-time implementation and ensures a consistent update rate regardless of the 
vehicle's motion. Nevertheless, it may lead to uneven spatial sampling of the control 
inputs, particularly when the vehicle speed changes significantly.
'''

from acados_template import AcadosOcpSolver, AcadosSimSolver, AcadosSim
from Solvers import *
from ReferenceGeneration import * 
from DomainProjection import *

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# VEHICLE PARAMETERS  - HAVE TO BE THE SAME AS IN THE MODELING  and SOLVERS
params = VehicleParams()
params.BoschCar() 


# ------------------ INITIAL CONFIGURATION ------------------

omega = 0.3 # NOTE: the value of omega very different for S CURVE and ELLIPSE
d_ocp = 0.05
d_sim = 0.05
dt_control = 0.02
N_horizon = 50
Tf =N_horizon * d_ocp

ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

y_ref_kin, y_ref_dyn, S, s_ref, kappa_ref= scurve.Spatial_Ref(omega, d_ocp, N_horizon) # depending on the model we choose the reference
X0 = y_ref_kin[:,0]
print(X0)
print(y_ref_kin.shape)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = int(S*d_ocp)#Nsim = int(S * d_ocp / d_sim)

time_solver = TimeSolvers(X0, N_horizon, Tf) # used for simulation 
spatial_solver = SpatialSolvers(s_ref, kappa, X0,N_horizon, Tf) # used for contol 

# ------------------ CLOSED LOOP SIMULATION------------------

def ClosedLoopSimulation():
    # Setup solvers
    ocp = spatial_solver.Solver_SpatialKin(d_ocp)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_spatial1.json')

    sim_solver =time_solver.Solver_NL_TimeKin(d_sim)
    sim = AcadosSim()
    sim.model = sim_solver.model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = d_sim  # [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_spatial1.json')

    # Simulation settings
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = []     # list of np.ndarray(nx,)
    simU = []     # list of np.ndarray(nu,)
    predX = []    # list of np.ndarray(nx,)
    predU = []    # list of np.ndarray(nu,)


    # Initialization
    y_ref = y_ref_kin  # this is since you are not using deltaFz
    xcurrent = y_ref[2:, 0 ]  # time-domain state [x, y, v, ψ]
    print(xcurrent)
    xcurr_ocp = X0.copy()    # spatial-domain state [eψ, ey, x, y, v, ψ]
    print(f"XCURR OCP:", xcurr_ocp)
    simX.append(xcurr_ocp.copy())
    predX.append(xcurr_ocp.copy())

    s_sim = s_ref[0]
    S_sim = np.array(s_sim)

    k = 0
    control_timer = 0.0
    u0 = np.zeros(nu)

    # Initialize OCP guesses
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurr_ocp)
        idx = min(stage, len(s_ref) - 1)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[idx]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.array([0.0, 0.0]))

    simulation_running = True
    i = 0

    # ---------------- MAIN LOOP ----------------
    while simulation_running:
        # Check if it’s time to compute control
        if control_timer >= dt_control - 1e-9:
            # Compute current s, eψ, ey from time-domain state
            s_now, epsi_now, ey_now = time2spatial(
                xcurrent[0], xcurrent[1], xcurrent[3], s_ref, y_ref[[2, 3, 5], :]
            )
  
            # Find nearest reference index
            idx_closest = int(np.argmin(np.abs(s_ref - s_now)))

            # Update horizon references
            for j in range(N_horizon):
                idx_stage = min(idx_closest + j, len(s_ref) - 1)
                yref_stage = np.concatenate(
                    (y_ref[:2, idx_stage], [0.0, 0.0])
                )
                acados_ocp_solver.set(j, "yref", yref_stage)
                acados_ocp_solver.set(j, "p", np.array([s_ref[idx_stage]]))

            # Terminal
            idx_term = min(idx_closest + N_horizon, len(s_ref) - 1)
            acados_ocp_solver.set(N_horizon, "yref", y_ref[:2, idx_term])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[idx_term]]))

            # Set initial state constraints for OCP
            x0_ocp = np.hstack((epsi_now, ey_now, xcurrent))
            acados_ocp_solver.set(0, "lbx", x0_ocp)
            acados_ocp_solver.set(0, "ubx", x0_ocp)
            acados_ocp_solver.set(0, "x", x0_ocp)

            # Solve OCP
            status = acados_ocp_solver.solve()
            if status != 0:
                print(f"[WARNING] ACADOS solver failed with status {status}")


            # Get optimal control
            u0 = acados_ocp_solver.get(0, "u").copy()
            control_timer -= dt_control  # reset control timer
            # Print progress
            print(f"s_sim = {s_sim:.3f}, idx_ref = {idx_closest}, control = {u0}")


        # -------- SIMULATION STEP --------
        simU.append(u0.copy())
        xnext = acados_sim_solver.simulate(xcurrent, u0, s_sim)
        xcurrent = xnext.copy()

        # Update spatial position and store logs
        s_sim, epsi, ey = time2spatial(
            xcurrent[0], xcurrent[1], xcurrent[3], s_ref, y_ref[[2, 3, 5], :]
        )
        
        x_logged = np.hstack((epsi, ey, xcurrent))
        x_logged[0] += np.arctan(params.lr * tan(u0[1]) / params.L)
        x_logged[-1] += np.arctan(params.lr * tan(u0[1]) / params.L)
        simX.append(x_logged)

        S_sim = np.append(S_sim, s_sim)

        # Update ocp initial guess for next iteration
        xcurr_ocp = np.hstack((epsi, ey, xcurrent))
        acados_ocp_solver.set(0, "lbx", xcurr_ocp)
        acados_ocp_solver.set(0, "ubx", xcurr_ocp)

        
        # Step forward
        i += 1
        control_timer += d_sim

        # Stop condition
        if s_sim >= (S - 2 * d_ocp):
            simulation_running = False
            print(f"stop condition fulfilled", S)
            print(f"s sim", s_sim)

    # ---------------- PLOTTING ----------------
    simX = np.asarray(simX)
    simU = np.asarray(simU)
    predX = np.asarray(predX)
    predU = np.asarray(predU)

    
    print(f"Number of iterations",i)
    t = np.linspace(0, (i + 1) * d_sim, i + 1)
    y_ref_time = reference_to_time(y_ref, t, S_sim)
    print(f"y ref time shape", y_ref_time.shape)
    print(f"SIM X SHAPE", simX.shape)
    plot_states(simX, simU, y_ref_time)

def reference_to_time(y_ref, time, s_sim):
    y_ref_time = np.zeros((len(time), y_ref.shape[0]))
    for j in range(y_ref.shape[0]):
        y_ref_time[:, j] = np.interp(
            s_sim,      # arc length values at each time step
            s_ref,              # arc length coordinates of reference points
            y_ref[j, :],        # reference values at those arc lengths
            left=y_ref[j, 0],   # value for points before first reference point
            right=y_ref[j,-1]  # value for points after last reference point
        )
    return y_ref_time


def plot_states(simX, simU, y_ref):
    Nf = simX.shape[0] - 1
    # Time vectors
    timestampsx = np.linspace(0, (Nf+1)*d_sim, Nf+1)
    timestampsu = np.linspace(0, Nf*d_sim, Nf)
    timestampsy = np.linspace(0, (N+1)*d_ocp, N+1)

    # --- Trajectory Plot ---
    plt.figure()
    plt.plot(simX[:Nf+1,2], simX[:Nf+1,3], label=r'Trajectory', linewidth=2)
    plt.plot(y_ref[:Nf+1,2], y_ref[:Nf+1,3], '--', alpha=0.9, label=r'Reference', color="C1", linewidth=1.5)
    plt.xlabel(r'$x [m]$')
    plt.ylabel(r'$y[m]$')
    #plt.axis('equal')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis('equal')

    # --- Subplot 1: Lateral Error ---

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(timestampsx, simX[:Nf+1,1], color="#808000", linewidth=2)
    axs[0].plot(timestampsx,  y_ref[:Nf+1, 1], '--', color="C1", linewidth=1.5)   
    axs[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
    axs[0].set_title("Lateral Tracking Error", fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 2: Heading Error ---
    axs[1].plot(timestampsx, simX[:Nf+1, 0],label=r'Simulation',  color="#808000", linewidth=2)
    axs[1].plot(timestampsx,  y_ref[:Nf+1, 0], '--', label=r'Reference', color="C1", linewidth=1.5)   
    axs[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
    axs[1].set_title("Heading Tracking Error", fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].set_xlabel(r'Time$[s]$')
    axs[1].legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    # --- 2. States ---
    state_indices = [2, 3, 4, 5]  # x, y, vx theta
    state_labels = [r'$x[m]$', r'$y[m]$', r'$v[\frac{m}{s}]$',  r'$\theta[rad]$']
    fig2, ax2 = plt.subplots(len(state_indices), 1, sharex=True, figsize=(6, 12))
    for j, idx in enumerate(state_indices):
        ax2[j].plot(timestampsx, simX[:Nf+1, idx],label = r'Simulation', color="C0", linewidth=2)
        ax2[j].plot(timestampsx, y_ref[:Nf+1, idx], '--',  label = r'Reference', color="#ffd166", linewidth=1.5)
        ax2[j].set_ylabel(state_labels[j])
        ax2[j].grid(True, linestyle="--", alpha=0.4)
    ax2[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax2[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax2[3].set_title("Direction Angle", fontsize=12, fontweight='bold')
    ax2[-1].set_xlabel(r'Time$[s]$')
    ax2[-1].legend(fontsize=11, loc='upper right')
    ax2[-1].grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    

    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    labels = [r'$a[\frac{m}{s^2}]$', r'$\delta[rad]$'] 
    
    ax2[0].plot(timestampsu, simU[:Nf, 0],color="C2", linewidth=2) #color="#A52A2A"
    ax2[0].set_ylabel(labels[0])
    ax2[0].set_title("Acceleration", fontsize=12, fontweight='bold')
    ax2[0].grid(True, linestyle="--", alpha=0.4)

    ax2[1].plot(timestampsu, simU[:Nf, 1],color="C3", linewidth=2) # color="#A52A2A"
    ax2[1].set_ylabel(labels[1])
    ax2[1].set_xlabel(r'Time$[s]$')
    ax2[1].set_title("Steering Angle", fontsize=12, fontweight='bold')
    ax2[1].grid(True, linestyle="--", alpha=0.6)

    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show(block=True)

    plt.show()


if __name__ == "__main__":
    ClosedLoopSimulation()