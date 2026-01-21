from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import patches
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant

from ref_time2spatial import *
from animation import SimulationAnimator
from half_spatial_kin_model import *

from reference_trajectory_dynamic import  *
from vehicle_params import  VehicleParams

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.1
dt_sim = 0.05
N_horizon = 10
Tf = N_horizon * ds_ocp

straight_line = StraightLineTrajectory()
ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

y_ref, S, s_full, kappa = scurve.spatial_ref(ds_ocp, N_horizon) # reference for the controller
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = int(S/ds_ocp) # number of iterations for controller
Nsim = 2*int(S/dt_sim)
print(y_ref[:, 0])
print(s_0[0])
X0 = y_ref[:, 0]
anim = SimulationAnimator(y_ref[:,2], y_ref[:,3], y_ref[:,4],ds_ocp, lf, lr)


params = VehicleParams()
params.RaceCar43()

# ------------------ MODELING ------------------

def TimeDynModel_RC():
    # States
    X   = SX.sym('X')
    Y  = SX.sym('Y')
    vx  = SX.sym('vx')
    vy  = SX.sym('vy')
    psi  = SX.sym('psi') 
    omega  = SX.sym('omega') # yaw rate - psi dot
    x = vertcat(X, Y, vx, vy,psi, omega)

    # Controls: acceleration a, motor characteristic
    delta = SX.sym('delta')
    D = SX.sym('D')
    u = vertcat (D, delta)

    X_dot = SX.sym("X_dot")
    Y_dot = SX.sym("Y_dot")
    vx_dot = SX.sym("vx_dot")
    vy_dot = SX.sym("vy_dot")
    psi_dot = SX.sym("psi_dot")
    omega_dot   = SX.sym('omega_dot')
    xdot = vertcat(X_dot, Y_dot, vx_dot, vy_dot, psi_dot, omega_dot)

    #beta = arctan(vy/(vx+1e-5)) # slipping angle
    beta_f = -arctan((vy + params.lf*omega)/(vx+1e-5)) + delta
    beta_r = arctan((params.lr*omega - vy)/(vx+1e-5))

    #LATERAL TIRE MODEL
    Fc_f = params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
    Fc_r =  params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r) )

    # SECOND ORDER FRICTION TERM
    Fx_d = (params.Cm1 -params.Cm2*vx)*D - params.cr2*vx**2 - params.cr0 #* tanh(params.cr3 *vx)

    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + (-Fx_d - Fc_f*sin(delta))/params.m
    dvy   = - vx*omega + (Fc_f*cos(delta) + Fc_r)/params.m
    dpsi = omega
    domega    = (params.lf*Fc_f*cos(delta) - params.lr*Fc_r)/params.I_z
    

    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega)
    model = AcadosModel()
    model.name        = 'DynamicRaceCar'
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.f_expl_expr = f_expl

    return model

def SpatialDynModel_RC():
    model_name = "SpatialDynamicBicycle_RC"

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
    D= SX.sym("D")
    delta = SX.sym("delta")
    u = vertcat(D, delta)

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
    beta_r = arctan(( params.lr*omega - vy)/(vx+1e-5))
    
    #LATERAL TIRE MODEL
    Fc_f = params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
    Fc_r =  params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r) )

    # SECOND ORDER FRICTION TERM
    Fx_d = (params.Cm1 -params.Cm2*vx)*D - params.cr2*vx**2 - params.cr0 #* tanh(params.cr3 *vx)

    Fyf = Fc_f*cos(delta) #+np.pi/2) # this is the same as -sin(delta)
    Fyr = Fc_r
    
    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + (-Fx_d - Fc_f*sin(delta))/params.m
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
    #f_impl = xdot - f_expl

    # algebraic variables
    z = vertcat([])
    # parameters
    p = vertcat(s)

    model = AcadosModel()
    #model.f_impl_expr = f_impl
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

# ------------------ACADOS SOLVER SETTINGS ------------------
def CreateOcpSolver_Spatial() -> AcadosOcp:
    ocp = AcadosOcp()
    model = SpatialDynModel_RC()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 2 + nu
    ny_e = 2

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e2,3*1e1])  
    R_mat =  np.diag([1e-1,1e-1])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ model.x[4]+1e-4), model.x[1], model.u) #
  
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = Q_mat*ds_ocp
    yref_e = np.array([0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[0]+ arctan(model.x[5]/ model.x[4]+1e-4), model.x[1]) 
 
    ocp.parameter_values  = np.array([s_ref[0]])
    # set constraints on the input                                                                                                             
    ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # constraints on the states
    ocp.constraints.lbx = np.array([ -np.deg2rad(20), -0.25])
    ocp.constraints.ubx = np.array([ np.deg2rad(20), 0.25])
    ocp.constraints.idxbx = np.array([0,1])

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 70
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = Tf
    return ocp


def CreateSimSolver_Time() -> AcadosOcp:
    ocp = AcadosOcp()
    model =TimeDynModel_RC()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([5*1e1, 5*1e2, 1e2, 1e0, 5*1e0,1e1])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = Q_mat#scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((nx,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/(model.x[2]+1e-5)), model.x[5] )#model.x #vertcat(model.x[:2], model.x[2]*cos(model.x[4])-model.x[3]*sin(model.x[4]), model.x[2]*sin(model.x[4])+model.x[3]*cos(model.x[4]), model.x[4]+arctan(model.x[3]/ (model.x[2]+1e-5)), model.x[-1])
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =Q_mat * dt_ocp#2 * np.diag([5*1e1, 5*1e1, 1e0, 1e0, 5*1e1,1e1]) *dt_ocp
    yref_e = np.array([2, 0.0, 0.0, 0.0, 0.0, 0.0]) 
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
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    return ocp


# ------------------CLOSED LOOP ------------------
def closed_loop_simulation():
    #AcadosOcpSovler
    ocp =CreateOcpSolver_Spatial() 
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_rt2.json')

    #AcadosIntegrator
    sim_solver= CreateSimSolver_Time()
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
    xcurrent = y_ref[2:,0]
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
                acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[:2, k+j],[0.0, 0.0])))
                acados_ocp_solver.set(j, "p", np.array([s_ref[k + j]]))
            acados_ocp_solver.set(N_horizon, "yref", y_ref[:2, k+N_horizon])
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
        xcurrent = acados_sim_solver.simulate(xcurrent, u0) 
        
        s_sim,epsi ,ey= time2spatial(xcurrent[0], xcurrent[1], xcurrent[2],s_ref,y_ref[:,2:5])
        simX[i + 1, :] = np.hstack((epsi,ey,xcurrent))
        simX[i + 1 , 0] =  simX[i + 1, 0]#+arctan (lr*tan(simU[i, 1])/L) # representation of epsi + Beta 
        simX[i + 1, -2] =simX[i + 1, -2] #+ arctan (lr*tan(simU[i, 1])/L) # representation of theta 
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
    anim.animate(simX[:i+1, 2],simX[:i+1, 3],simX[:i+1, 4],simX[:i+1, 5],simU[:i+1, 1],t)  
    

def plot_states(simX, simU, y_ref_time,Nf):
    timestampsx = np.linspace(0,(Nf+1)*dt_sim,Nf+1)
    timestampsu = np.linspace(0,(Nf)*dt_sim,Nf)
    timestampsy = np.linspace(0,(N+1)*ds_ocp,N+1)
    Ny = timestampsy.shape[0]

    plt.figure()
    plt.plot(simX[:Nf+1,2],simX[:Nf+1,3], label='Simulation')
    plt.plot(y_ref[:Ny+1,2], y_ref[:Ny+1,3], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y", "x", "y", "theta", "v", "a", "delta"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:Nf+1, i])
        ax[i].plot(timestampsx, y_ref[:Nf+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:Nf, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
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
