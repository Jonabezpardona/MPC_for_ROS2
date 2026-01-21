'''
TIME KINEMATIC BICYCLE MODEL WITH DIFFERENT SAMPLING TIME FOR CONTROLLER AND SIMULATOR

SAMPLING TIMES OF CONTROLLER AND SIMULATOR ARE DIFFERENT
'''
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import time
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from casadi import SX, vertcat, sin, cos, tan, arctan
from reference_trajectory_kinematic import *


# ------------------ VEHICLE PARAMETERS ------------------

lf = 0.1335#2.3 # distance from CoG to front wheels
lr = 0.1335#1.2 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
omega = 0.3
dt_ocp = 0.05
dt_sim = 0.05
N_horizon = 20
Tf =N_horizon * dt_ocp

ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

trajectory, T = scurve.time_ref(dt_ocp, N_horizon)
X0 = trajectory[:,0]

N = int(T /dt_ocp)
Nsim = int(T / dt_sim)

# ------------------ MODEL and OCP SOLVER ------------------


def Time_KinematiBicycleModel():
    model_name = "time_kin_bicycle"
    # STATES AND CONTROL
    # SX is scalar symbolic type
    x = SX.sym("x")
    y = SX.sym("y")
    v = SX.sym("v")
    psi = SX.sym("psi")
    x = vertcat(x, y, v, psi) # vertically concatenate
    a = SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)
    beta = arctan(lr * tan(delta) / L) # slipping angle
    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    v_dot = SX.sym("v_dot")
    psi_dot = SX.sym("psi_dot")
    xdot = vertcat(x_dot, y_dot, v_dot, psi_dot)
    # dynamics
    f_expl = vertcat(v * cos(psi+beta), v * sin(psi+beta), a ,v *sin(beta) /lr) 
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name = model_name
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$v$", "$\\psi$"]
    model.u_labels = [ "$a$", "$delta$"]
    return model



def CreateOcpSolver_TimeKin() -> AcadosOcp: #NONLINEAR SOLVER
    ocp = AcadosOcp()
    model = Time_KinematiBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    #Q_mat = 2 * np.diag([1e2,1e2, 1e3, 1e2])  #ellipse
    Q_mat = 2*np.diag([1e1, 1e1 , 1e1, 1e1])  #s_surve

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((nx,)) #ny
    ocp.model.cost_y_expr = vertcat( model.x[:3], model.x[3]+arctan(lr*tan(model.u[1])/L) )
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.diag([1e1, 1e1 , 1e1])*dt_ocp
    #yref_e = np.array([10.0, 0.0, 0.0]) #ellipse path
    yref_e = np.array([50.0, 0.0, 0.0])  #s curve path
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:3]) 
  
    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([ -3, -np.deg2rad(60)])
    ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
    ocp.solver_options.tf = Tf
    #ocp.solver_options.qp_solver_cond_N =   # MUST BE INTEGER, ONLY FOR PARTIAL CONDENSING
    return ocp

def CreateLINOcpSolver_TimeKin() -> AcadosOcp: # LINEAR SOLVER
    ocp = AcadosOcp()
    model = time_kinematic_bicycle_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    #Q_mat = 2 * np.diag([1e2,1e2, 1e3, 1e2])  #ellipse
    Q_mat =  np.diag([1e2, 1e2 , 1e3, 1e2])  #s_surve

    #path const
    ocp.cost.cost_type = "LINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,)) 
    #terminal cost    Q_mat = 2 * np.diag([1e2, 1e2, 0.0, 0.0, 0.0,0.0])  

    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e =2 * np.diag([1e1, 1e1 , 1e1, 1e1])*dt_ocp
    #yref_e = np.array([10.0, 0.0, 0.0, np.pi/2]) #ellipse path
    yref_e = np.array([50.0, 0.0, 0.0, 0.0])  #s curve path
    ocp.cost.yref_e = yref_e
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([ -3, -np.deg2rad(60)])
    ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0
    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
    ocp.solver_options.tf = Tf
    #ocp.solver_options.qp_solver_cond_N =   # MUST BE INTEGER, ONLY FOR PARTIAL CONDENSING
    return ocp


# ----------------- CLOSED LOOP SIMULATION ------------------

def ClosedLoopSimulationSIM():# LOOP OVER N SIM, UPDATING N OCP ACCORDINGLY
    ocp = CreateOcpSolver_TimeKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_nonlinear.json')

    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK'  
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps =1 
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_nonlinear.json')
    
    #simulation
    N_horizon = acados_ocp_solver.N
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))                                                  
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((N_horizon+ 1, nx))
    predU = np.zeros((N_horizon+1, nu))
    yref_= np.zeros((N+N_horizon+1,nx))
    xcurrent = X0
    simX[0, :] = xcurrent
    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))
    for stage in range(N_horizon+N+1):
        yref_[stage, :] = trajectory[:, stage] #np.concatenate((trajectory[:, stage], [0.0, 0.0]))
        
    for i in range(Nsim):
    # update reference
        k = int(dt_ocp/dt_sim)
        if  ((i%k)==0):
            for j in range(N_horizon): 
                acados_ocp_solver.set(j, "yref", trajectory[:, i//k+j]) #int(i/k)
            acados_ocp_solver.set(N_horizon, "yref", trajectory[:3, i//k+N_horizon]) #int(i/k)
        status = acados_ocp_solver.solve()
        if status != 0 and status != 2:
            print(f"ACADOS solver failed with status {status}")
        #x0 = acados_ocp_solver.get(0, "x")
        u0 = acados_ocp_solver.get(0, "u")
        
        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        #xcurrent = Integrator(xcurrent, u0)
        simX[i+ 1, :] = xcurrent
        simU[i, :] = u0
        '''
        stats = {
            "time_tot": acados_ocp_solver.get_stats("time_tot"),
            "time_lin": acados_ocp_solver.get_stats("time_tot"),
            "qp_iter": acados_ocp_solver.get_stats("qp_iter"),
            "res_stat": acados_ocp_solver.get_stats("residuals"),
            #"res_eq": acados_ocp_solver.get_stats("res_eq_all"),
            #"res_ineq": acados_ocp_solver.get_stats("res_ineq_all"),
        }
        print(stats)
        '''
        # update initial condition
        x0 = acados_ocp_solver.get(1, "x")
        acados_ocp_solver.set(0, "lbx", xcurrent) 
        acados_ocp_solver.set(0, "ubx", xcurrent) 
        print('x' , xcurrent)
        print('y', simU[i,:])

    plot_trajectory(simX, simU, yref_)


def ClosedLoopSimulationOCP():
    print("\n--- Starting Closed-Loop Simulation (OCP-based) ---")
    ocp = CreateOcpSolver_TimeKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_linear.json')

    # Simulation model (plant)
    sim = AcadosSim()
    sim.model = ocp.model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = int(dt_ocp / dt_sim)
    sim.solver_options.T = dt_sim
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_linear.json')

    # Setup arrays
    N_horizon = acados_ocp_solver.N
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    yref_ = np.zeros((N + N_horizon + 1, nx))

    xcurrent = X0
    simX[0, :] = xcurrent

    # Initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros(nu))
    for stage in range(N_horizon + N + 1):
        yref_[stage, :] = trajectory[:, stage]

    # --- Performance tracking ---
    solve_times = []
    loop_times = []
    qp_iterations = []

    # --- Closed-loop simulation ---
    for i in range(N):
        loop_start = time.time()

        # Update reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", trajectory[:, i + j])
        acados_ocp_solver.set(N_horizon, "yref", trajectory[:3, i + N_horizon])

        # --- Solve MPC ---
        t0 = time.time()
        status = acados_ocp_solver.solve()
        t1 = time.time()

        if status not in [0, 2]:
            print(f"⚠️ ACADOS solver failed at step {i} with status {status}")

        # --- Record solver stats ---
        solve_time = acados_ocp_solver.get_stats("time_tot")

        qp_iter_stats = acados_ocp_solver.get_stats("qp_iter")
        # take last iteration if array, otherwise use scalar
        if hasattr(qp_iter_stats, "__len__"):
            qp_iter = qp_iter_stats[-1]
        else:
            qp_iter = qp_iter_stats

        solve_times.append(solve_time)
        qp_iterations.append(qp_iter)

        # --- Apply control ---
        u0 = acados_ocp_solver.get(0, "u")

        for k in range(int(dt_ocp / dt_sim)):
            idx = i * int(dt_ocp / dt_sim) + k
            xcurrent = acados_sim_solver.simulate(xcurrent, u0)
            simX[idx + 1, :] = xcurrent
            simU[idx, :] = u0

        # --- Update for next iteration ---
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        loop_end = time.time()
        loop_times.append(loop_end - loop_start)

        print(f"Step {i:03d} | Solve time: {solve_time*1000:.2f} ms | "
              f"Loop time: {(loop_end - loop_start)*1000:.2f} ms | "
              f"QP iter: {qp_iter}")

    # --- Convert to arrays ---
    solve_times = np.array(solve_times, dtype=float)
    loop_times = np.array(loop_times, dtype=float)
    qp_iterations = np.array(qp_iterations, dtype=float)

    # --- Performance summary ---
    print("\n========== MPC PERFORMANCE SUMMARY ==========")
    print(f"Average MPC solve time: {np.mean(solve_times)*1000:.3f} ms")
    print(f"Max MPC solve time: {np.max(solve_times)*1000:.3f} ms")
    print(f"Min MPC solve time: {np.min(solve_times)*1000:.3f} ms")
    print(f"Average control loop time: {np.mean(loop_times)*1000:.3f} ms")
    print(f"Total simulation time: {np.sum(loop_times):.3f} s")
    print(f"Average QP iterations: {np.mean(qp_iterations):.4f}")
    print(f"Real-time ratio (loop_time / dt_ocp): {np.mean(loop_times)/dt_ocp:.2f}")
    print("=============================================\n")

    # --- Optional: Plot timing traces ---
    plt.figure(figsize=(8, 4))
    plt.plot(np.array(solve_times) * 1000, label="Solve time [ms]")
    plt.plot(np.array(loop_times) * 1000, label="Loop time [ms]")
    plt.xlabel("Iteration")
    plt.ylabel("Time [ms]")
    plt.title("MPC Timing Performance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Plot vehicle trajectory and control inputs
    plot_trajectory(simX, simU, yref_)

# ----------------- PLOTTING ------------------
def plot_trajectory(simX, simU, yref_):
    timestampsx = np.linspace(0,(Nsim+1)*dt_sim,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*dt_sim,Nsim)
    timestampsy = np.linspace(0,(N+1)*dt_ocp,N+1)

    fig0 = plt.figure(figsize=(8, 6))
    plt.plot(simX[:,0],simX[:,1],label = r'Simulation', color = "C0", linewidth=2)
    plt.plot(yref_[:N+1,0], yref_[:N+1,1], '--', label = r'Reference', alpha=0.9 , color="C1", linewidth=1.5)
    plt.xlabel(r'$x[m]$')
    plt.ylabel(r'$y[m]$')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis('equal')

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 14))
    #fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = [r'$x[m]$', r'$y[m]$', r'$v[\frac{m}{s}]$', r'$\theta[rad]$']
    ulabels = [r'Simulation', r'Reference']
    titles = ["X Coordinates", "Y Coordinates", "Speed", "Direction Angle"]
    for i in range(3):
        ax[i].plot(timestampsx, simX[:, i], linewidth=2) #, color="#3498DB"
        ax[i].plot(timestampsy, yref_[:N+1,i], '--', color="C1", linewidth=1.5)
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True, linestyle="--", alpha=0.4)
    ax[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax[2].set_title("Speed", fontsize=12, fontweight='bold')
    theta =  simX[:-1,3] +  arctan(lr * tan(simU[:,1]) / L) # define it better
    ax[3].plot(timestampsu, theta, label = r'Simulation' , color="#3498DB", linewidth=2) # simX[:,3]
    ax[3].plot(timestampsy, yref_[:N+1,3], '--', label = r'Reference' ,  color="C1", linewidth=1.5)
    ax[3].set_ylabel(labels[3])
    ax[3].set_title(titles[3], fontsize=12, fontweight='bold')
    ax[3].set_xlabel(r'Time $[s]$')
    ax[3].legend(fontsize=11, loc='upper right')
    ax[3].grid(True, linestyle="--", alpha=0.4)
    #handles = [Line2D([0], [0]), #, color="#3498DB"
    #           Line2D([0], [0], linestyle='--',  color="C1")]
    #fig.legend(handles, ulabels, loc='lower right', ncol=1)
    #fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    labels = [r'$a[\frac{m}{s^2}]$', r'$\delta[rad]$'] 
    
    ax2[0].plot(timestampsu, simU[:, 0],color="C2", linewidth=2) #color="#A52A2A"
    ax2[0].set_ylabel(labels[0])
    ax2[0].set_title("Acceleration", fontsize=12, fontweight='bold')
    ax2[0].grid(True, linestyle="--", alpha=0.4)

    ax2[1].plot(timestampsu, simU[:, 1],color="C3", linewidth=2) # color="#A52A2A"
    ax2[1].set_ylabel(labels[1])
    ax2[1].set_xlabel(r'Time $[s]$')
    ax2[1].set_title("Steering Angle", fontsize=12, fontweight='bold')
    ax2[1].grid(True, linestyle="--", alpha=0.6)

    fig2.tight_layout(rect=[0, 0.05, 1, 1])




    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ClosedLoopSimulationOCP()
    

