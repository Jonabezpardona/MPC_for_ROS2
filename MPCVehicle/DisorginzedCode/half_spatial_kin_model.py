from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant
from reference_trajectory_spatial import  *
#from reference_trajectory_generation import  *
from RK4integration import RK4_HalfSpatKinBic

# ------------------ VEHICLE PARAMS ------------------
# some values, should be a script for itself
lf = 0.1335 # distance from CoG to front wheels
lr = 0.1335 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
dt_ocp = 0.05
dt_sim = 0.05
N_horizon = 20
Tf =N_horizon *dt_ocp

#traight_line = StraightLineTrajectory()
ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()
#track = TrajectoryGeneration()
#y_ref, N, s_ref, kappa_ref = track.spatial_reference(N_horizon)
S, s_0, s_ref, kappa_ref, y_ref = ellipse.halfspatial_offline_ref(dt_ocp, dt_sim, N_horizon)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)
X0 = y_ref[0,:]
#S = N*0.1
N = int(S/dt_ocp) 
Nsim = int(S/dt_sim)

#rk4 = RK4_HalfSpatKinBic(lr,L,kappa)

# ------------------ MODELING ------------------
def HalfSpatial_KinematiBicycleModel():
    model_name = "HalfSpatialKinBicycleModel"
    ## CasADi Model
    s = SX.sym('s')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v= SX.sym("v")
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    x = vertcat(e_psi, e_y, x, y, psi, v, s)

    # Controls: steering angle delta, acceleration a
    a= SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    v_dot = SX.sym("v_dot")
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    s_dot = SX.sym("s_dot")
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,v_dot,s_dot)

    beta = arctan(lr * tan(delta) / L) # slipping angle
    vx = v* cos(psi+beta)
    vy = v* sin(psi+beta)
    dpsi = v *sin(beta) / lr
    
    #Spatial dynamics dx/ds = f(x,u)
    sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta)*sin(e_psi))/(1 - kappa(s)* e_y)
    dx_ds    = vx 
    dy_ds    = vy 
    dv_ds    = a 
    dpsi_ds  = (dpsi) 
    d_e_psi = (dpsi) - kappa(s)*sdot
    d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) 

    f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, dv_ds, sdot)
    f_impl = xdot - f_expl

    # algebraic variables
    z = vertcat([])
    # parameters
    p = vertcat([]) 

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u 
    model.p = p
    model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$v$", "$s$"]
    model.u_labels = [ "$a$", "$delta$"]
    model.name = model_name
    return model


def HalfSpatial_KinematiBicycleModel1():
    model_name = "HalfSpatialKinBicycleModel1"
    ## CasADi Model
    s = SX.sym('s')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v= SX.sym("v")
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    x = vertcat(e_psi, e_y, x, y, psi, s)

    # Controls: steering angle delta, acceleration a
    #a= SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(v, delta)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    v_dot = SX.sym("v_dot")
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    s_dot = SX.sym("s_dot")
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,s_dot)

    beta = arctan(lr * tan(delta) / L) # slipping angle
    vx = v* cos(psi+beta)
    vy = v* sin(psi+beta)
    dpsi = v *sin(beta) / lr
    
    #Spatial dynamics dx/ds = f(x,u)
    sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta)*sin(e_psi))/(1 - kappa(s)* e_y)
    dx_ds    = vx 
    dy_ds    = vy 
    #dv_ds    = a 
    dpsi_ds  = (dpsi) 
    d_e_psi = (dpsi) - kappa(s)*sdot
    d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) 

    f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, sdot)
    f_impl = xdot - f_expl

    # algebraic variables
    z = vertcat([])
    # parameters
    p = vertcat([]) 

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u 
    model.p = p
    model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$s$"]
    model.u_labels = [  "$v$",  "$delta$"]
    model.name = model_name
    return model

# ------------------ACADOS SOLVER SETTINGS ------------------
def CreateOcpSolver_HalfSpatialKin() -> AcadosOcp:
    ocp = AcadosOcp()
    model = HalfSpatial_KinematiBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 3 + nu
    ny_e = 3

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e1,5*1e1,1e2]) #1e3,5*1e2,1e5 
    R_mat =  np.diag([1e-1,1e-2]) #1e-1,1e-3

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.x[-1], model.u)
    
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = Q_mat*dt_ocp
    yref_e = np.array([0.0, 0.0, 7.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:2], model.x[-1]) 

    # set constraints  - this is for the input                                                                                                             
    ocp.constraints.lbu = np.array([-1.5, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1.5, np.deg2rad(30)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # constraints on the states
    ocp.constraints.lbx = np.array([ -np.deg2rad(20), -0.5])
    ocp.constraints.ubx = np.array([ np.deg2rad(20), 0.5])
    ocp.constraints.idxbx = np.array([0,1])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 70
    ocp.solver_options.tf = Tf
    return ocp


# ------------------CLOSED LOOP ------------------
def ClosedLoopSimulation():
    #AcadosOcpSovler
    ocp = CreateOcpSolver_HalfSpatialKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_spatialtiral.json')

    #AcadosIntegrator
    
    sim = AcadosSim()
    sim.model = ocp.model   
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_spatialtrial.json')

    #simulation
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim +N_horizon+ 1, nx))
    predU = np.zeros((Nsim+N_horizon, nu))
    #initialization
    xcurrent = X0
    print('x0', X0)
    predX[0,:] = xcurrent
    simX[0, :] = xcurrent


    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0])) #warm start
    
    for i in range(N):
        # Reference for horizon
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[j+i,:2],[y_ref[j+i,-1]],[0.0, 0.0])))
        acados_ocp_solver.set(N_horizon, "yref", np.concatenate((y_ref[i+N_horizon,:2],[y_ref[i+N_horizon,-1]])))

        # SOLVE OCP PROBLEM
        status = acados_ocp_solver.solve()
        if status != 0:
            print(f"ACADOS solver failed with status {status}")
        
        for j in range(N_horizon):
            predX[i+j,:] = acados_ocp_solver.get(j, "x")
            predU[i+j,:] = acados_ocp_solver.get(j, "u")
        predX[i+N_horizon,:] = acados_ocp_solver.get(N_horizon, "x")
        
        # update initial condition - move to the next state
        u0 = acados_ocp_solver.get(0, "u")
        xcurrent = acados_sim_solver.simulate(xcurrent, u0) 
        #xcurrent = rk4.FullIntegrator(xcurrent, u0, dt_sim)
        simU[i, :] = u0
        simX[i + 1, :] = xcurrent
        simX[i + 1 , 0] = xcurrent[0] +arctan (lr*tan(simU[i, 1])/L) # representation of epsi 
        simX[i + 1, -3] = xcurrent[-3] +arctan (lr*tan(simU[i, 1])/L) # representation of theta 

        
        acados_ocp_solver.set(0, "lbx",xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
        '''
        simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
        xcurrent = acados_sim_solver.simulate(xcurrent, simU[i,:])
        simX[i + 1, :] = xcurrent
        print(xcurrent)
        '''
        print('x' , xcurrent)
        print('y',y_ref[i+1,:])
        print('u', simU[i,:])
    plot_trajectory(simX, simU, y_ref)
    

def plot_trajectory(simX, simU, y_ref):
    timestampsx = np.linspace(0,(Nsim+1)*dt_sim,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*dt_sim,Nsim)
    timestampsy = np.linspace(0,(N+1)*dt_ocp,N+1)

    fig0 = plt.figure(figsize=(8, 6))
    plt.plot(simX[:,2],simX[:,3], label=r'Simulation', linewidth=2)
    plt.plot(y_ref[:N+1,2], y_ref[:N+1,3], '--', alpha=0.9 , c = "orange" ,label=r'Reference', linewidth=1.5 )
    plt.xlabel(r'$x[m]$')
    plt.ylabel(r'$y[m]$')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis('equal')


    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    # --- Subplot 1: Lateral Error ---
    axs[0].plot(timestampsx, simX[:,1], color="#808000", linewidth=2)
    axs[0].plot(timestampsy, y_ref[:N+1,1], '--', color="C1", linewidth=1.5)   
    axs[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
    axs[0].set_title("Lateral Tracking Error", fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 2: Heading Error ---
    axs[1].plot(timestampsx, simX[:, 0], color="#808000", linewidth=2)
    axs[1].plot(timestampsy, y_ref[:N+1,1], '--', color="C1", linewidth=1.5)   
    axs[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
    axs[1].set_title("Heading Tracking Error", fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 3: Travelled Distance ---
    axs[2].plot(timestampsx, simX[:, -1], label=r'Simulation', color="#808000", linewidth=2)
    axs[2].plot(timestampsy, y_ref[:N+1,-1], '--', color="C1", label = r'Reference', linewidth=1.5)   
    axs[2].set_xlabel(r'Time$[s]$', fontsize=12)
    axs[2].set_ylabel(r'$s \; [\mathrm{m}]$', fontsize=12)
    axs[2].set_title("Travelled Distance", fontsize=12, fontweight='bold')
    axs[2].legend(fontsize=11, loc='lower right')
    axs[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()


    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 14))
    labels = [r'$x[m]$', r'$y[m]$', r'$v[\frac{m}{s}]$', r'$\theta[rad]$']
    for i in range(2):
        ax[i].plot(timestampsx, simX[:, i+2], linewidth=2) #, color="#3498DB"
        ax[i].plot(timestampsy, y_ref[:N+1,i+2], '--', color="C1", linewidth=1.5)
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True, linestyle="--", alpha=0.4)
    ax[2].plot(timestampsx, simX[:, 5], linewidth=2) #, color="#3498DB"
    ax[2].plot(timestampsy, y_ref[:N+1,5], '--', color="C1", linewidth=1.5)
    ax[2].set_ylabel(labels[2])
    ax[2].grid(True, linestyle="--", alpha=0.4)
    theta =  simX[:-1,4] +  arctan(lr * tan(simU[:,1]) / L) # define it better
    ax[3].plot(timestampsu, simX[:-1,4], label = r'Simulation' , color="#3498DB", linewidth=2) # simX[:,3]
    ax[3].plot(timestampsy, y_ref[:N+1,4], '--', label = r'Reference' ,  color="C1", linewidth=1.5)
    ax[3].set_ylabel(labels[3])
    ax[3].set_xlabel(r'Time$[s]$')
    ax[3].legend(fontsize=11, loc='upper right')
    ax[3].grid(True, linestyle="--", alpha=0.4)

    ax[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax[3].set_title("Direction Angle", fontsize=12, fontweight='bold')
    
    
    
    
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
    plt.show()



if __name__ == "__main__":
    ClosedLoopSimulation()
