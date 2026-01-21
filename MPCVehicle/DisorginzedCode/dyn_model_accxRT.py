from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant, fabs
from reference_trajectory_dynamic import *
from ref_time2spatial import *
from vehicle_params import VehicleParams

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.01
dt_sim = 0.05
N_horizon = 50
Tf =N_horizon * ds_ocp

straight_line = StraightLineTrajectory()
ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

y_ref, S, s_ref, kappa_ref= scurve.spatial_ref( ds_ocp, N_horizon)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)
N = int(S/ds_ocp) 
Nsim =5* int(S/dt_sim)
X0 = y_ref[0, :]

params = VehicleParams()
params.BoschCar()  

# ------------------ MODELING ------------------
def Spatial_DynamicBicycleModel():
    model_name = "SpatialDynamicBicycle_model"

    ## CasADi Model
    s = SX.sym('s')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    vx  = SX.sym('vx')
    vy  = SX.sym('vy')
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    omega  = SX.sym('r')
    x = vertcat(e_psi, e_y, x, y, vx, vy, psi, omega)

    # Controls: steering angle delta, acceleration a
    a= SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)

    # xdot
    s_dot = SX.sym("s_dot")
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    vx_dot = SX.sym("vx_dot")
    vy_dot = SX.sym("vy_dot")
    psi_dot = SX.sym("psi_dot")
    omega_dot   = SX.sym('omega_dot')
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot,vx_dot, vy_dot, psi_dot,omega_dot)

    beta = arctan(vy/(vx+1e-5)) # slipping angle
    # Slip angles  -  aprox of small angles
    beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta #+np.pi/2
    beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))
    
    Fx_d = 1/2*params.ro* params.Cz* params.Az* vx*fabs(vx)
    # Lateral tire forces - simplified Pacejka # forget about them, try 
    Fc_f = -params.mi *params.Dcf * sin( params.Ccf * arctan(1/params.mi *params.Bcf * beta_f) )
    Fc_r = - params.mi * params.Dcr * sin( params.Ccr * arctan(1/params.mi * params.Bcr * beta_r) )

    Fyf = Fc_f*cos(delta) #+np.pi/2) # this is the same as -sin(delta)
    Fyr = Fc_r
    
    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + a +  (-Fx_d - Fc_f*sin(delta))/params.m# we consider that this is completly longitudial acceleration
    dvy   = - vx*omega+ (Fyf + Fyr)/params.m
    dpsi = omega
    domega    = (params.lf*Fyf - params.lr*Fyr)/params.I_z


    #Spatial dynamics dx/ds = f(x,u)
    sdot = (vx * cos(e_psi) - vy *sin(e_psi))/(1 - kappa(s)* e_y)
    
    dx_ds    = dX / (sdot)
    dy_ds    = dY / (sdot)
    dvx_ds    = dvx/ (sdot)
    dvy_ds    = dvy/ (sdot)
    dpsi_ds  = (dpsi) / (sdot)
    domega_ds  = (domega) / (sdot)
    d_e_psi = (dpsi)/(sdot) - kappa(s)
    d_e_y = (vx  * sin(e_psi) + vy* cos(e_psi)) / (sdot)
    f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds,dvx_ds, dvy_ds, dpsi_ds, domega_ds)
    f_impl = xdot - f_expl

    # algebraic variables
    z = vertcat([])
    # parameters
    p = vertcat(s)

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u 
    model.p = p
    model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$",  "$v_x$",  "$v_y$", "$\\psi$", "$omega$"]
    model.u_labels = [ "$a$", "$delta$"]
    model.p_labels    = ["$s$"] 
    model.name = model_name
    return model


def TimeDynModel_accx():
    # States
    x   = SX.sym('x')
    y   = SX.sym('y')
    vx  = SX.sym('vx')
    vy  = SX.sym('vy')
    psi = SX.sym('psi')
    omega  = SX.sym('r') # yaw rate - psi dot
    x = vertcat(x, y, vx, vy, psi, omega)

    # Controls: acceleration a, steering delta
    delta = SX.sym('delta')
    a = SX.sym('a')
    u = vertcat (a, delta)
    
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    vx_dot = SX.sym("vx_dot")
    vy_dot = SX.sym("vy_dot")
    psi_dot = SX.sym("psi_dot")
    omega_dot   = SX.sym('omega_dot')
    xdot = vertcat(x_dot, y_dot, vx_dot, vy_dot, psi_dot, omega_dot)

    beta = arctan(vy/(vx+1e-5)) # slipping angle
    # Slip angles  -  aprox of small angles
    beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta #+np.pi/2
    beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))
    
    Fx_d = 1/2*params.ro* params.Cz* params.Az* vx*fabs(vx)
    # Lateral tire forces - simplified Pacejka # forget about them, try 
    Fc_f = -params.mi *params.Dcf * sin( params.Ccf * arctan(1/params.mi *params.Bcf * beta_f) )
    Fc_r = -params.mi*params.Dcr * sin( params.Ccr * arctan(1/params.mi * params.Bcr * beta_r) )

    Fyf = Fc_f*cos(delta) #+np.pi/2) # this is the same as -sin(delta)
    Fyr = Fc_r


    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + a  +  (-Fx_d - Fc_f*sin(delta))/params.m# we consider that this is completly longitudial acceleration
    dvy   = - vx*omega+ (Fyf + Fyr)/params.m
    dpsi = omega
    domega    = (params.lf*Fyf - params.lr*Fyr)/params.I_z


    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega)
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name        = 'time_dyn_bicyc_a'
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model

# ------------------ACADOS SOLVER SETTINGS ------------------
def CreateOcpSolver_SpatialDyn() -> AcadosOcp:
    ocp = AcadosOcp()
    model = Spatial_DynamicBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 3 + nu
    ny_e = 3

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e2,5*1e1,1e0])  
    R_mat =  np.diag([1e-1,1e-1])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1],model.x[4], model.u) #
  
    #terminal costs
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    Q_mat = np.diag([1e2,5*1e1])
    ocp.cost.W_e = Q_mat*ds_ocp
    yref_e = np.array([0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1]) 
 
    ocp.parameter_values  = np.array([s_ref[0]])
    # set constraints on the input                                                                                                             
    ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
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
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = Tf
    return ocp


def CreateOcpSolver_TimeDyn() -> AcadosSim:
    ocp = AcadosOcp()
    model = TimeDynModel_accx()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([5*1e3, 5*1e3, 5*1e2, 5*1e-1, 5*1e1,5*1e2,1e-1,1e0]) #np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e2])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/(model.x[2]+1e-5)), model.x[5] , model.u)
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([5*1e1, 5*1e0, 1e1, 1e1, 5*1e1,1e1]) *ds_ocp
    yref_e = np.array([10, 0.0, 0.0, 0.0, np.pi/2, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = model.x #vertcat(model.x[:4]) 

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([-5,-np.deg2rad(45)])
    ocp.constraints.ubu = np.array([5, np.deg2rad(45)])
    ocp.constraints.idxbu = np.array([0,1])
    ocp.constraints.x0 = X0
    #ocp.constraints.lbx = np.array([-10, -10])
    #ocp.constraints.ubx = np.array([10, 10])
    #ocp.constraints.idxbx = np.array([2,3])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    return ocp

# ------------------CLOSED LOOP ------------------
def ClosedLoopSimulation():
    #AcadosOcpSovler
    ocp = CreateOcpSolver_SpatialDyn()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_spatial1.json')

    #AcadosIntegrator
    sim_solver = CreateOcpSolver_TimeDyn()
    sim = AcadosSim()
    sim.model = sim_solver.model   
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_spatial1.json')

    #simulation
    simulaton_running = True
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    Beta = np.zeros((Nsim, 1))
    predX = np.zeros((Nsim +N_horizon+ 1, nx))
    predU = np.zeros((Nsim+N_horizon, nu))

    #initialization
    xcurrent = y_ref[0, 2:]
    xcurr_ocp = X0
    print('x0', X0)
    predX[0,:] = xcurr_ocp
    simX[0, :] = xcurr_ocp
    s_prev = - ds_ocp
    s_sim = s_ref[0]
    S_sim = np.array(s_sim)
    k = 0

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurr_ocp)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[stage]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0])) #warm start
    
    simulation_running = True
    i = 0
    while simulation_running: 
        # Reference for horizon
        if (s_sim- s_prev >= ds_ocp) or(s_sim >=s_ref[k-1]):
            for j in range(N_horizon):
                acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[j+k,:2],[y_ref[j+k,4]], [0.0, 0.0])))
                acados_ocp_solver.set(j, "p", np.array([s_ref[k + j]]))
            acados_ocp_solver.set(N_horizon, "yref", y_ref[k+N_horizon, :2])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[k + N_horizon]]))
            s_prev = s_sim
            k = k+1
            
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

        simU[i, :] = u0
        xprev = xcurrent
        xcurrent = acados_sim_solver.simulate(xcurrent, u0, s_sim) 
        s_sim,epsi ,ey= time2spatial(xcurrent[0], xcurrent[1], xcurrent[4],s_ref,y_ref[:, [2,3,6]])
        
        simX[i + 1, :] = np.hstack((epsi,ey,xcurrent))
        simX[i + 1 , 0] = simX[i + 1 , 0] + np.arctan(simX[i + 1 , 5]/ (simX[i + 1 , 4])) # representation of epsi, it only affects what we are seeing on the graph, not on interation itself.
        simX[i + 1, -2] = simX[i + 1, -2] + np.arctan(simX[i + 1 , 5]/ (simX[i + 1 , 4])) # representation of theta 
        #s_sim,_ ,_= time2spatial(xcurrent[2], xcurrent[3], xcurrent[6],s_ref,y_ref[[2,3,6],:])
        #s_sim = s_sim + np.sqrt((xcurrent[2]-xprev[2])**2+(xcurrent[3]-xprev[3])**2)
        S_sim = np.append(S_sim, s_sim)
        xcurr_ocp = np.hstack((epsi,ey,xcurrent))
        acados_ocp_solver.set(0, "lbx",xcurr_ocp)
        acados_ocp_solver.set(0, "ubx", xcurr_ocp)

        # prints
        print('SREF' , s_ref[k])
        print('Ssim' , s_sim)
        #print('S' ,s_0[-1])
        #print('x' , xcurrent)
        #print('y',y_ref[:,k+1])
        #print('u', simU[i,:])
        i = i+1
        if (s_sim>= (S- 2*ds_ocp)):
            simulation_running= False 
    t = np.linspace(0,(i+1)*dt_sim,i+1)

    y_ref_time  = reference_to_time(y_ref, t, S_sim)
    plot_states(simX, simU, y_ref_time, i)

def reference_to_time(y_ref, time, s_sim):
    y_ref_time = np.zeros((len(time), y_ref.shape[1]))
    for j in range(y_ref.shape[1]):
        y_ref_time[:, j] = np.interp(
            s_sim,      # arc length values at each time step
            s_ref,              # arc length coordinates of reference points
            y_ref[:, j],        # reference values at those arc lengths
            left=y_ref[0, j],   # value for points before first reference point
            right=y_ref[-1, j]  # value for points after last reference point
        )
    return y_ref_time


def plot_states(simX, simU, y_ref,Nf):
    # Time vectors
    timestampsx = np.linspace(0, (Nf+1)*dt_sim, Nf+1)
    timestampsu = np.linspace(0, Nf*dt_sim, Nf)
    timestampsy = np.linspace(0, (N+1)*ds_ocp, N+1)

    # --- Trajectory Plot ---
    plt.figure()
    plt.plot(simX[:Nf+1,2], simX[:Nf+1,3], label=r'Trajectory', linewidth=1.7)
    plt.plot(y_ref[:Nf+1,2], y_ref[:Nf+1,3], '--', alpha=0.9, label=r'Reference', color="#ffd166", linewidth=1.2)
    plt.xlabel(r'$x [m]$')
    plt.ylabel(r'$y[m]$')
    #plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    # --- 1. Errors ---
    fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
    error_labels = [r'$e_{\psi}[rad]$', r'$e_y$[m]']
    for i in range(2):
        ax1[i].plot(timestampsx, simX[:Nf+1, i], color="#1ABC9C", linewidth=1.7 )
        ax1[i].plot(timestampsx, y_ref[:Nf+1,i], '--',  color="#ffd166", linewidth=1.2)
        ax1[i].set_ylabel(error_labels[i])
    ax1[-1].set_xlabel(r'travelled distance[m]')
    handles = [Line2D([0], [0], color="#1ABC9C"),
               Line2D([0], [0], linestyle='--',  color="#ffd166")]
    labels = [r'Simulation', r'Reference']
    fig1.legend(handles, labels, loc='lower center', ncol=2)
    fig1.tight_layout(rect=[0, 0.05, 1, 1])

    # --- 2. States ---
    state_indices = [2, 3, 4, 5, 6, 7]  # x, y, vx, vy, theta, omega
    state_labels = [r'$x[m]$', r'$y[m]$', r'$v_x[\frac{m}{s}]$', r'$v_y[\frac{m}{s}]$', r'$\theta[rad]$', r'$\omega[\frac{rad}{s}]$']
    fig2, ax2 = plt.subplots(len(state_indices), 1, sharex=True, figsize=(6, 8))
    for j, idx in enumerate(state_indices):
        ax2[j].plot(timestampsx, simX[:Nf+1, idx], color="#3498DB", linewidth=1.7)
        ax2[j].plot(timestampsx, y_ref[:Nf+1, idx], '--',  color="#ffd166", linewidth=1.2)
        ax2[j].set_ylabel(state_labels[j])
    ax2[-1].set_xlabel(r'travelled distance [m]')
    handles = [Line2D([0], [0], color="#3498DB"),
               Line2D([0], [0], linestyle='--',  color="#ffd166")]
    fig2.legend(handles, labels, loc='lower center', ncol=2)
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    

    # --- 3. Controls ---
    control_labels = [r'$a[\frac{m}{s^2}]$', r'$\delta[rad]$']
    fig3, ax3 = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
    for k in range(2):
        ax3[k].plot(timestampsu, simU[:Nf, k],color="#A52A2A", linewidth=1.7)
        ax3[k].set_ylabel(control_labels[k])
    ax3[-1].set_xlabel(r'travelled distance[m]')
    handles_ctrl = [Line2D([0], [0], color="#A52A2A")]
    ctrl_labels = [ r'Simulation']
    fig3.legend(handles_ctrl, ctrl_labels, loc='lower center', ncol=1)
    fig3.tight_layout(rect=[0, 0.05, 1, 1])

    plt.show()
if __name__ == "__main__":
    ClosedLoopSimulation()
