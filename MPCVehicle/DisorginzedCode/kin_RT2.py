from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import patches
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant

from ref_time2spatial import *
from animation import SimulationAnimator
from half_spatial_kin_model import *

from reference_trajectory_spatial import  *

# ------------------ VEHICLE PARAMS ------------------
# some values, should be a script for itself
lf = 0.1335 # distance from CoG to front wheels
lr = 0.1335 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.05
dt_sim = 0.01
N_horizon = 30
Tf = N_horizon * ds_ocp

straight_line = StraightLineTrajectory()
ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

S, s_0, s_ref, kappa_ref, y_ref = scurve.spatial_offline_ref(ds_ocp, N_horizon) # reference for the controller
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = int(S/ds_ocp) # number of iterations for controller
Nsim = 6*int(S/dt_sim)
print(y_ref[0,:])
print(s_0[0])
X0 = y_ref[0, :]
#anim = SimulationAnimator(y_ref[:,2], y_ref[:,3], y_ref[:,4],ds_ocp, lf, lr)

# ------------------ MODELING ------------------
def spatial_bicycle_model_ocp():
    model_name = "SpatialBicycleOCP"

    ## CasADi Model
    s = SX.sym('s')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v= SX.sym("v")
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    x = vertcat(e_psi, e_y, x, y, psi, v)

    # Controls: steering angle delta, acceleration a
    a= SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)

    # xdot
    s_dot = SX.sym("s_dot")
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    v_dot = SX.sym("v_dot")
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,v_dot)

    beta = arctan(lr * tan(delta) / L) # slipping angle
    vx = v* cos(psi+beta)
    vy = v* sin(psi+beta)
    dpsi = v *sin(beta) / lr
    
    #Spatial dynamics dx/ds = f(x,u)
    sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta) *sin(e_psi))/(1 - kappa(s)* e_y)
    dx_ds    = vx / (sdot)
    dy_ds    = vy / (sdot)
    dv_ds    = a / (sdot)
    dpsi_ds  = (dpsi) / (sdot)
    d_e_psi = (dpsi)/(sdot) - kappa(s)
    d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) / (sdot)
    f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, dv_ds)
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
    model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$v$"]
    model.u_labels = [ "$a$", "$delta$"]
    model.p_labels    = ["$s$"] 
    model.name = model_name
    return model

def time_bicycle_model_sim():
    model_name = "TimeBicycleSim"

    # SX is scalar symbolic type
    x = SX.sym("x")
    y = SX.sym("y")
    v = SX.sym("v")
    psi = SX.sym("psi")
    x = vertcat(x, y, psi, v) # vertically concatenate
    a = SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)
    beta = arctan(lr * tan(delta) / L) # slipping angle
    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    v_dot = SX.sym("v_dot")
    psi_dot = SX.sym("psi_dot")
    xdot = vertcat(x_dot, y_dot, psi_dot, v_dot)
    # dynamics
    f_expl = vertcat(v * cos(psi+beta), v * sin(psi+beta),v *sin(beta) /lr, a) 
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name = model_name
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$\\psi$", "$v$"]
    model.u_labels = [ "$a$", "$delta$"]
    return model

# ------------------ACADOS SOLVER SETTINGS ------------------
def create_ocp_solver() -> AcadosOcp:
    ocp = AcadosOcp()
    model = spatial_bicycle_model_ocp()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 2 + nu
    ny_e = 2

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e3,5*1e2])  
    R_mat =  np.diag([1e0,1e-3])

    ocp.parameter_values  = np.array([s_ref[0]])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.u) #
  
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = Q_mat*ds_ocp
    yref_e = np.array([0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:2]) 

    # set constraints  - this is for the input                                                                                                             
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

def create_sim_solver() -> AcadosOcp:
    ocp = AcadosOcp()
    model = time_bicycle_model_sim()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([1e1,1e1, 1e1, 5*1e2])  #ellipse
    #Q_mat = 2 * np.diag([1e2, 1e2 , 1e3, 1e2])  #s_surve

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((nx,)) #ny
    ocp.model.cost_y_expr = vertcat( model.x[:2], model.x[2]+arctan(lr*tan(model.u[1])/L) , model.x[3])
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([1e2, 1e2 , 1e1])*dt_sim
    yref_e = np.array([10.0, 0.0, 0.0]) #ellipse path
    #yref_e = np.array([50.0, 0.0, 0.0])  #s curve path
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:3]) 
  
    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([ -1, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
    ocp.solver_options.tf = Tf
    return ocp



# ------------------INTEGATION BY HAND ERK4------------------
def one_step_interation(ey, epsi, psi, v, a, beta):
    sdot = (v *cos(beta) * cos(epsi) - v *sin(beta) *sin(epsi))/(1 - kappa(scurrent)* ey)
    dx =  v* cos(psi+beta)/ sdot
    dy =  v * sin(psi+beta)/ sdot
    dv =  a/ sdot
    dpsi = v *sin(beta) / lr / sdot
    depsi = dpsi - kappa(scurrent) 
    dey = (v *cos(beta) * sin(epsi) + v *sin(beta) * cos(epsi)) / sdot
    return depsi,dey,dx,dy,dpsi,dv 

def Integrator(xcurrent, u, scurrent):
    print("Integration by ERK4")
    h = 0.5

    a = u[0]
    beta = arctan(lr * tan(u[1]) / L) 
    epsi = xcurrent[0]
    ey = xcurrent[1]
    x = xcurrent[2]
    y = xcurrent[3]
    psi = xcurrent[4]
    v  = xcurrent[5]
    
    depsi1,dey1,dx1,dy1,dpsi1,dv1 = one_step_interation(ey, epsi, psi, v, a, beta)
    depsi2,dey2,dx2,dy2,dpsi2,dv2 = one_step_interation(ey+ h *ds_sim *dey1,epsi+ h *ds_sim *depsi1 , psi + h *ds_sim *dpsi1, v + h *ds_sim *dv1, a, beta)
    depsi3,dey3,dx3,dy3,dpsi3,dv3 = one_step_interation(ey+ h *ds_sim *dey2, epsi+ h *ds_sim *depsi2, psi + h *ds_sim *dpsi2, v + h *ds_sim *dv2, a, beta)
    depsi4,dey4,dx4,dy4,dpsi4,dv4 = one_step_interation(ey+ h *ds_sim *dey3, epsi+ ds_sim *depsi3, psi + ds_sim *dpsi3, v + ds_sim *dv3, a, beta)
    #next
    epsi_next = epsi + ds_sim * (depsi1+2*depsi2+2*depsi3+depsi4)/6
    ey_next = ey + ds_sim * (dey1+2*dey2+2*dey3+dey4)/6
    x_next = x + ds_sim * (dx1+2*dx2+2*dx3+dx4)/6
    y_next = y + ds_sim * (dy1+2*dy2+2*dy3+dy4)/6
    v_next = v + ds_sim * (dv1+2*dv2+2*dv3+dv4)/ 6
    psi_next = psi + ds_sim * (dpsi1+2*dpsi2+2*dpsi3+dpsi4)/ 6
    x_current = np.stack((epsi_next,ey_next, x_next,y_next,v_next,psi_next))

    return x_current

# ------------------CLOSED LOOP ------------------
def closed_loop_simulation():
    #AcadosOcpSovler
    ocp = create_ocp_solver()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_rt2.json')

    #AcadosIntegrator
    sim_solver= create_sim_solver()
    sim = AcadosSim()
    sim.model = sim_solver.model   
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_rt2.json') # instead of it we can use EKR that we made

    #simulation
    simulaton_running = True
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim +N_horizon+ 1, nx))
    predU = np.zeros((Nsim+N_horizon, nu))

    #initialization 
    xcurrent = y_ref[0, 2:]
    print(xcurrent)
    xcurr_ocp = X0
    predX[0,:] = xcurr_ocp
    simX[0, :] = xcurr_ocp
    s_sim = s_0[0]
    S_sim = np.array(s_sim)
    s_prev = - ds_ocp
    k = 0

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurr_ocp)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[stage]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.1])) #warm start

    simulation_running = True
    i = 0
    while simulation_running: 
        # Reference for horizon
        # compute ssim - what is the arc of length in simulation
        #s_sim = s_sim + dt_sim * xcurrent[-1]
        if (s_sim- s_prev >= ds_ocp) or (s_sim > s_ref[k+1]):
            for j in range(N_horizon):
                acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[j+k,:2],[0.0, 0.0])))
                acados_ocp_solver.set(j, "p", np.array([s_ref[k + j]]))
            acados_ocp_solver.set(N_horizon, "yref", y_ref[k+N_horizon,:2])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[k+ N_horizon]]))
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
            # get the control
            u0 = acados_ocp_solver.get(0, "u")
            
        simU[i, :] = u0
        #acados_ocp_solver.set("p", np.array([s_ref[k+1]]))
        xcurrent = acados_sim_solver.simulate(xcurrent, u0) 
        
        s_sim,epsi ,ey= time2spatial(xcurrent[0], xcurrent[1], xcurrent[2],s_ref,y_ref[:,2:5])
        simX[i + 1, :] = np.hstack((epsi,ey,xcurrent))
        simX[i + 1 , 0] =  simX[i + 1, 0]+arctan (lr*tan(simU[i, 1])/L) # representation of epsi + Beta 
        simX[i + 1, -2] =simX[i + 1, -2] + arctan (lr*tan(simU[i, 1])/L) # representation of theta 
        S_sim = np.append(S_sim, s_sim)
        xcurr_ocp = np.hstack((epsi,ey,xcurrent))
        acados_ocp_solver.set(0, "lbx",xcurr_ocp)
        acados_ocp_solver.set(0, "ubx", xcurr_ocp)
        
        #anim.plot_car_RT(xcurrent[0], xcurrent[1], xcurrent[2], xcurrent[3], u0[1],i*dt_sim)
        # it was anim.plot_bicycle() Find that funtion!!!
        # prints
        print('SREF' , s_ref[k])
        print('Ssim' , s_sim)
        print('S' ,s_0[-1])
        '''
        print('x sim' , xcurrent)
        print('x ocp' , xcurr_ocp)
        print('y',y_ref[k,:])
        print('u', simU[i,:])  
        '''
        i = i+1
        if (s_sim>= (S- 2*ds_ocp)):
            simulation_running= False 
    t = np.linspace(0,(i+1)*dt_sim,i+1)

    y_ref_time  = reference_to_time(y_ref, t, S_sim)
    plot_states(simX, simU, y_ref_time, i)
    #anim.animate(simX[:i+1, 2],simX[:i+1, 3],simX[:i+1, 4],simX[:i+1, 5],simU[:i+1, 1],t) 
     
    

def plot_states(simX, simU, y_ref_time,Nf):
    timestampsx = np.linspace(0,(Nf+1)*dt_sim,Nf+1)
    timestampsu = np.linspace(0,(Nf)*dt_sim,Nf)
    timestampsy = np.linspace(0,(N+1)*ds_ocp,N+1)
    Ny = timestampsy.shape[0]

    plt.figure(figsize=(8, 6))
    plt.plot(simX[:Nf+1,2],simX[:Nf+1,3], label=r'Simulation', linewidth = 2)
    plt.plot(y_ref_time[:,2], y_ref_time[:,3], '--', alpha=0.9 , c = "orange" ,label=r'Reference', linewidth = 1.5)
    plt.xlabel(r'$x[m]$')
    plt.ylabel(r'$y[m]$')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis('equal')

    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Subplot 1: Lateral Error ---
    axs[0].plot(timestampsx, simX[:Nf+1,1], color="#808000", linewidth=2)
    axs[0].plot(timestampsx,  y_ref_time[:Nf+1, 1], '--', color="C1", linewidth=1.5)   
    axs[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
    axs[0].set_title("Lateral Tracking Error", fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 2: Heading Error ---
    axs[1].plot(timestampsx, simX[:Nf+1, 0],label=r'Simulation',  color="#808000", linewidth=2)
    axs[1].plot(timestampsx,  y_ref_time[:Nf+1, 0], '--', label=r'Reference', color="C1", linewidth=1.5)   
    axs[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
    axs[1].set_title("Heading Tracking Error", fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].set_xlabel(r'Time$[s]$')
    axs[1].legend(fontsize=11, loc='lower right')
    plt.tight_layout()


    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8,14))
    labels = [r'$x[m]$', r'$y[m]$', r'$v[\frac{m}{s}]$', r'$\theta[rad]$']
    for j in range(2):
        ax[j].plot(timestampsx, simX[:Nf+1, j+2], linewidth = 2)
        ax[j].plot(timestampsx,  y_ref_time[:Nf+1, j+2], '--', label='Reference', color = "C1", linewidth = 1.5)
        ax[j].set_ylabel(labels[j])
        ax[j].grid(True, linestyle="--", alpha=0.4)
    ax[2].plot(timestampsx, simX[:Nf+1, 5], linewidth=2) #, color="#3498DB"
    ax[2].plot(timestampsx, y_ref_time[:Nf+1,5], '--', color="C1", linewidth=1.5)
    ax[2].set_ylabel(labels[2])
    ax[2].grid(True, linestyle="--", alpha=0.4)
    theta =  simX[:Nf+1,4] +  arctan(lr * tan(simU[:Nf+1,1]) / L) # define it better
    ax[3].plot(timestampsx, simX[:Nf+1,4], label = r'Simulation' , color="#3498DB", linewidth=2) # simX[:,3]
    ax[3].plot(timestampsx, y_ref_time[:Nf+1,4], '--', label = r'Reference' ,  color="C1", linewidth=1.5)
    ax[3].set_ylabel(labels[3])
    ax[3].set_xlabel(r'Time$[s]$')
    ax[3].legend(fontsize=11, loc='upper right')
    ax[3].grid(True, linestyle="--", alpha=0.4)

    ax[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax[3].set_title("Direction Angle", fontsize=12, fontweight='bold')
    
    
    
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


if __name__ == "__main__":
    closed_loop_simulation()
