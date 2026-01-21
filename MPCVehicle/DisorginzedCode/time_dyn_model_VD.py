from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan,  tanh, fabs
from reference_trajectory_dynamic import *
from vehicle_params import VehicleParams
 
# ------------------ INITIAL CONFIGURATION ------------------
#omega_ref = 0.03 # SCURVE
omega_ref = 0.3 #ELLIPSE
dt_ocp = 0.05
dt_sim = 0.05
N_horizon = 50
Tf = N_horizon * dt_ocp


ellipse = EllipseTrajectory()
straight_line = StraightLineTrajectory()
scurve = SCurveTrajectory()
trajectory,X0 , T = ellipse.time_ref(omega_ref, dt_ocp, N_horizon)
#X0 = np.hstack((X0, [0.0]))

N = int(T /dt_ocp)
Nsim = int(T / dt_sim)
a_max = 1

params = VehicleParams()
params.JaguarX()

# DOUBLE BICYCLE MODEL - VIRTUAL DRIVER FOR HIGH PERFORMANCE
def time_dynamical_bicycle_model_normalforces(): 
    # States
    x   = SX.sym('x')
    y   = SX.sym('y')
    vx  = SX.sym('vx')
    vy  = SX.sym('vy')
    psi = SX.sym('psi')
    omega  = SX.sym('omega') # yaw rate - psi dot
    delta_Fz = SX.sym('delta_Fz')
    x = vertcat(x, y, vx, vy, psi, omega, delta_Fz)

    # Controls: acceleration a, steering delta
    # may be needing normalied acc/breaking coeficient !!!!!
    delta = SX.sym('delta')
    a = SX.sym('a')
    u = vertcat (a, delta)
    
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    vx_dot = SX.sym("vx_dot")
    vy_dot = SX.sym("vy_dot")
    psi_dot = SX.sym("psi_dot")
    omega_dot   = SX.sym('omega_dot')
    delta_Fz_dot = SX.sym('delta_Fz_dot')
    xdot = vertcat(x_dot, y_dot, vx_dot, vy_dot, psi_dot, omega_dot, delta_Fz_dot)

    vy_f =  vy + params.lf * omega 
    vy_r = vy - params.lr * omega  #TODO: ja mislim da je ovde minus
    vx_f = vx 
    vx_r = vx 

    # velocities for each wheel frame, we take it as if they are the same for left and right wheel on the same axis (front and rear)
    # cornering - lateral
    vc_f= vy_f*cos(delta)- vx_f*sin(delta) 
    vc_r =vy_r
    # longitudial
    vl_f= vy_f*sin(delta) + vx_f*cos(delta) 
    vl_r =vx_r
    
    # Slip angles on tires
    beta_f = arctan(vc_f/(vl_f+1e-5)) # usually it is with delta!!!
    beta_r = arctan(vc_r/(vl_r+1e-5))
    
    Fx_d = 0.5*params.ro* params.Cz* params.Az* vx*fabs(vx)#air drag force

    # Lateral tire forces - simplified Pacejka 
    Fc_f = -params.mi *params.Dcf * sin( params.Ccf * arctan(1/params.mi *params.Bcf * beta_f) )  
    Fc_r = -params.mi *params.Dcr * sin( params.Ccr * arctan(1/params.mi *params.Bcr * beta_r) )

    Fzf_0 = params.m * params.g * params.lr / (params.L) 
    Fzr_0 = params.m * params.g * params.lf / (params.L) 
      # this is if my calculations are correct

    Fzf  = Fzf_0 - delta_Fz
    Fzr  = Fzr_0 + delta_Fz

    # Longitudial tire forces - simplified Pacejka 
    gamma = tanh(a / a_max) 
    Fl_f = gamma * params.mi * Fzf
    Fl_r = gamma * params.mi * Fzr

    #Fc_f = Fc_r = Fl_f = Fl_r = 0
    # LONGITUDIAL WHEEL FORCES
    Fx_f = Fl_f*cos(delta)- Fc_f*sin(delta) # FRONT LEFT
    Fx_r = Fl_r #REAR RIGHT
    Fx = Fx_f + Fx_r #2*Fx_f + 2* Fx_r 

    # LATERAL WHEEL FORCES
    Fy_f = Fl_f*sin(delta)  + Fc_f*cos(delta) # FRONT
    Fy_r = Fc_r # REAR 
    #Fyf = 2*Fy_f 
    #Fyr = 2*Fx_rl

    


    deltaz = -params.h *Fx/(2*params.L)
    tau_z = 0.5

    # FULL DYNAMICS MODEL
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + (Fx - Fx_d)/params.m
    dvy   = - vx*omega +  (Fy_f + Fy_r)/params.m
    dpsi = omega
    domega    = (params.lf*Fy_f - params.lr*Fy_r)/params.I_z
    ddelta_Fz = (deltaz - delta_Fz)/tau_z


    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega, ddelta_Fz)
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name        = 'time_bicyc_dynamical_model_deltaFz'
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model

def create_ocp_solver_BodyFrame() -> AcadosOcp:
    ocp = AcadosOcp()
    model = time_dynamical_bicycle_model_normalforces()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu - 1
    ny_e = nx-1
    ocp.solver_options.N_horizon = N_horizon
    Q_mat =  np.diag([1e1, 1e2, 1e2, 1e1, 1e-1,1e-2,1e-1, 1e-1]) #np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e2])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4] +arctan(model.x[3]/(model.x[2]+1e-5)) , model.x[5] , model.u)
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =np.diag([1e1, 1e2, 1e2, 1e-1, 1e1,1e-1]) *dt_ocp
    yref_e = np.array([20, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = model.x[:-1]  

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([-a_max,-np.deg2rad(45)])
    ocp.constraints.ubu = np.array([a_max, np.deg2rad(45)])
    ocp.constraints.idxbu = np.array([0,1])
    ocp.constraints.x0 = X0


    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    return ocp


def create_ocp_solver_BodyFrame1() -> AcadosOcp:
    ocp = AcadosOcp()
    model = time_dynamical_bicycle_model_normalforces()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu 
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e1, 1e2, 1e2, 1e1, 1e1,1e-1,1e-1, 1e-1,1e-1]) #np.diag([5*1e2, 5*1e1, 5*1e1, 5*1e1, 5*1e2,5*1e4,1e1, 1e0, 1e-1]) #np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e2])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4], model.x[5], model.x[6] , model.u)
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.diag([1e1, 1e2, 1e2, 1e1, 1e1,1e-1,1e-1]) *dt_ocp#np.diag([5*1e1, 5*1e1, 1e-1, 1e-1, 5*1e2,1e4, 1e1]) *dt_ocp
    yref_e = np.array([20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = model.x  

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([-a_max,-np.deg2rad(45)])
    ocp.constraints.ubu = np.array([a_max, np.deg2rad(45)])
    ocp.constraints.idxbu = np.array([0,1])
    ocp.constraints.x0 = X0


    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    return ocp


def closed_loop_simulationBodyFrame():
    ocp = create_ocp_solver_BodyFrame1()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_dyn_nl.json')
    
    sim = AcadosSim()
    sim.model = ocp.model    
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_dyn_nl.json')
    
    #simulation
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    yref_= np.zeros((Nsim+N_horizon,nx))

    xcurrent = X0
    simX[0, :] = xcurrent

    # initialize solver
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([ 0.0, 0.0]))
    for stage in range(N_horizon+Nsim):
        yref_[stage, :] = trajectory[:, stage]
        
    for i in range(N):
    # update reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((trajectory[:, j+i],[0.0,0.1])))
        acados_ocp_solver.set(N_horizon, "yref", trajectory[:, i+ N_horizon])
      
        status = acados_ocp_solver.solve()
        if status != 0:
            print(f"ACADOS solver failed with status {status}")
        u0 = acados_ocp_solver.get(0, "u")
        
        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        simU[i, :] = u0
        simX[i+ 1, :] = xcurrent

        # update initial condition
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
        
        print('x' , xcurrent)
        print('y' , trajectory[:, i+1])
        print('u', simU[i,:])
    plot_trajectory(simX, simU, yref_)
    
 

def plot_trajectory(simX, simU, yref_):
    timestampsx = np.linspace(0,(Nsim+1)*dt_sim,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*dt_sim,Nsim)
    timestampsy = np.linspace(0,(N+1)*dt_ocp,N+1)

    plt.figure()
    plt.plot(simX[:,0],simX[:,1], label='Simulation')
    plt.plot(yref_[:N+1,0], yref_[:N+1,1], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

    nx = simX.shape[1]-1
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu+1, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["x", "y", "vx", "vy", "theta", "omega", "a", "delta", "deltaFz"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsy, yref_[:N+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].plot(timestampsx, simX[:, -1])
    ax[-1].set_ylabel(labels[-1])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    closed_loop_simulationBodyFrame()

