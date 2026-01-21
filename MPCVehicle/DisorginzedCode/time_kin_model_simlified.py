'''
TIME KINEMATIC BICYCLE MODEL 
'''
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan
from reference_trajectory_kinematic import *


# ------------------ VEHICLE PARAMETERS ------------------

lf = 1.3#2.3 # distance from CoG to front wheels
lr = 1.3#1.2 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
omega = 0.3
dt = 0.1
N_horizon = 20
Tf =N_horizon * dt

ellipse = EllipseTrajectory()
trajectory, T = ellipse.time_ref(omega, dt, N_horizon)
X0 = trajectory[:,0]

N = int(T /dt)
Nsim = int(T / dt)

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

def CreateOcpSolver() -> AcadosOcp: 
    ocp = AcadosOcp()
    model = Time_KinematiBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon

    #ellipse
    Q_mat = 2 * np.diag([1e2, 1e2, 1e1, 5*1e2])  
    R_mat = np.diag([1e-1, 1e-1])

    #s_surve
    #Q_mat = 2 * np.diag([1e2, 1e2 , 1e5, 1e1])  
    #R_mat = 2 * 5 * np.diag([1e1, 1e-1])

    #path const
    ocp.cost.cost_type = "LINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat) # ,  R_mat)
    ocp.cost.yref = np.zeros((ny,)) #nu

    #terminal cost
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e =2 * np.diag([1e2, 1e2 , 1e1, 5*1e2])*dt
    yref_e = np.array([10.0, 0.0, 0.0, np.pi/2]) #ellipse path
    #yref_e = np.array([50.0, 0.0, 0.0, 0.0])  #s curve path
    ocp.cost.yref_e = yref_e
    
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([ -1, -np.deg2rad(60)])
    ocp.constraints.ubu = np.array([1, np.deg2rad(60)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0
    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
    ocp.solver_options.tf = Tf
    return ocp


# ----------------- CLOSED LOOP SIMULATION ------------------

def ClosedLoopSimulation():# LOOP OVER N SIM, UPDATING N OCP ACCORDINGLY
    ocp = CreateOcpSolver()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_nonlinear.json')

    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK'  
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps =1 
    sim.solver_options.T = dt # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_nonlinear.json')
    
    #simulation
    N_horizon = acados_ocp_solver.N
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))                                                  
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((N_horizon+ 1, nx))
    predU = np.zeros((N_horizon+1, nu))
    yref_= np.zeros((N+N_horizon+1,nx+nu))
    xcurrent = X0
    simX[0, :] = xcurrent
    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.1]))
    for stage in range(N_horizon+N+1):
        yref_[stage, :] = np.concatenate((trajectory[:, stage], [0.0, 0.5]))
        
    for i in range(N):
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", yref_[i+j, :])
        acados_ocp_solver.set(N_horizon, "yref", yref_[i+j, :4])
        
        status = acados_ocp_solver.solve()
        if status != 0 and status != 2:
            print(f"ACADOS solver failed with status {status}")
        #x0 = acados_ocp_solver.get(0, "x")
        u0 = acados_ocp_solver.get(0, "u")
        
        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        simX[i+ 1, :] = xcurrent
        simU[i, :] = u0

        for j in range(N_horizon):
            predX[j,:] = acados_ocp_solver.get(j, "x")
            predU[j,:] = acados_ocp_solver.get(j, "u")
        predX[N_horizon,:] = acados_ocp_solver.get(N_horizon, "x")


        # update initial condition
        x0 = acados_ocp_solver.get(1, "x")
        acados_ocp_solver.set(0, "lbx", xcurrent) 
        acados_ocp_solver.set(0, "ubx", xcurrent) 
        print('x' , xcurrent)
        print('y', simU[i,:])

    plot_trajectory(simX, simU, yref_)

# ----------------- PLOTTING ------------------
def plot_trajectory(simX, simU, yref_):
    timestampsx = np.linspace(0,(Nsim+1)*dt,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*dt,Nsim)
    timestampsy = np.linspace(0,(N+1)*dt,N+1)

    plt.figure()
    plt.plot(simX[:,0],simX[:,1], label='Simulation')
    plt.plot(yref_[:N+1,0], yref_[:N+1,1], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["x", "y", "v", "heading+slipping angle", "a", "steering angle"]
    for i in range(3):
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsy, yref_[:N+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    theta =  simX[:-1,3] +  arctan(lr * tan(simU[:,1]) / L) # define it better
    ax[3].plot(timestampsu, theta) # simX[:,3]
    ax[3].plot(timestampsy, yref_[:N+1,3], '--', label='Reference')
    ax[3].set_ylabel(labels[3])
    for i in range(2):
        ax[i + 4].plot(timestampsu, simU[:, i])
        ax[i + 4].set_ylabel(labels[i + 4])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ClosedLoopSimulation()
    

