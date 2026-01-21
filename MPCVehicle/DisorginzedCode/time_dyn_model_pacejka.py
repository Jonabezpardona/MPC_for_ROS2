from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan,  tanh
from copy import deepcopy
from reference_trajectory_dynamic import *
from vehicle_params import VehicleParams

# ------------------ INITIAL CONFIGURATION ------------------
omega_ref = 0.1
dt_ocp = 0.1
dt_sim = 0.1


N_horizon = 30
Tf = N_horizon * dt_ocp


ellipse = EllipseTrajectory()
straight = StraightLineTrajectory()
scurve  = SCurveTrajectory()

trajectory,X0 , T = scurve.time_ref(omega_ref, dt_ocp, N_horizon)


N = int(T /dt_ocp)
Nsim = int(T / dt_sim)

params = VehicleParams()
params.JaguarX()  

# does not work 
# BOTH LATERAL AND LONGITUDIAL FORCES ARE BASED ON PACEJKA + TODO: look at the models in the book more carefully
# aproximated friction coeficient based on the acceleration
def TimeDynBicyclePacejka():
    # States
    x   = SX.sym('x')
    y   = SX.sym('y')
    vx  = SX.sym('vx')
    vy  = SX.sym('vy')
    psi = SX.sym('psi')
    omega  = SX.sym('omega') # yaw rate - psi dot
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


    beta = arctan((vy)/(vx+1e-5))
    # Slip angles ON TIRES - aprox of small angles
    beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta
    beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))

    vf = vx +params.rr * params.lf* beta 
    vr = vx -params.rr * params.lr * beta 
    
    kf = (omega* params.rr - vf)/(vf+1e-5)#(vy)/(vx+1e-5)# 
    kr = (omega* params.rr - vr)/(vr+1e-5)

    Fx_d = 1/2*params.ro* params.Cz* params.Az* vx**2

    mi  = 0.5

    params.mi = mi + (a+1e-5)/params.g

    # Lateral tire forces - simplified Pacejka 
    Fc_f = -params.mi * params.Dcf * sin( params.Ccf * arctan(1/params.mi * params.Bcf * beta_f) )
    Fc_r = -params.mi *params.Dcr * sin( params.Ccr * arctan( 1/params.mi *params.Bcr * beta_r) )

    # Longitudinal tire forces - simplified Pacejka 
    Fl_f = params.mi *params.Dlf * sin( params.Clf * arctan(1/params.mi *params.Blf * kf) )
    Fl_r = params.mi *params.Dlr * sin( params.Clr * arctan(1/params.mi *params.Blr * kr) )
    
    Fxf = Fl_f*cos(delta)- Fc_f*sin(delta)
    Fxr = Fl_r
    Fyf = Fl_f*sin(delta)  + Fc_f*cos(delta)
    Fyr = Fc_r

    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + ( - Fx_d +Fxf + Fxr)/params.m
    dvy   = - vx*omega + (Fyf + Fyr)/params.m
    dpsi = omega
    domega    = (params.lf*Fyf - params.lr*Fyr)/params.I_z


    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega)
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name        = 'time_dynamical_model_pacejka_forces'
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model


# ACADOS SOLVER SETTINGS
def create_ocp_solver_WorldFrame() -> AcadosOcp:
    ocp = AcadosOcp()
    model = TimeDynBicyclePacejka()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([5*1e0, 5*1e0, 1e1, 1e0, 5*1e1,1e2,1e-1,1e0])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:2], model.x[2]*cos(model.x[4])-model.x[3]*sin(model.x[4]), model.x[2]*sin(model.x[4])+model.x[3]*cos(model.x[4]), model.x[4]+arctan(model.x[3]/ model.x[2]+1e-5), model.x[5], model.u)
    #+arctan(model.x[3]/model.x[2])
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e1]) *dt_ocp
    yref_e = np.array([10, 0.0, 0.0, 0.0, np.pi/2, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = model.x #vertcat(model.x[:4]) 

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([-5,-0.8])#-np.deg2rad(45)
    ocp.constraints.ubu = np.array([5, 0.8])#np.deg2rad(45)
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

def closed_loop_simulationWorlFrame():
    ocp = create_ocp_solver_WorldFrame()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_dyn_pcb.json')
    
    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    #sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_dyn_pcb.json')
    
    #simulation
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    yref_= np.zeros((Nsim+N_horizon,nx))
    xcurrent = X0
    simX[0, :] = xcurrent
    simX[0, 2] = xcurrent[2]*cos(xcurrent[4])-xcurrent[3]*sin(xcurrent[4])
    simX[0, 3] = xcurrent[2]*sin(xcurrent[4])+xcurrent[3]*cos(xcurrent[4])
    simX[0, -2] = xcurrent[-2]+ arctan(xcurrent[3]/ xcurrent[2])# representation of theta 
        

    # initialize solver
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))
    for stage in range(N_horizon+Nsim):
        yref_[stage, :] = trajectory[:, stage]
        
    for i in range(N):
    # update reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((trajectory[:, j+i],[0,0])))
        acados_ocp_solver.set(N_horizon, "yref", trajectory[:,i+N_horizon])
      
        status = acados_ocp_solver.solve()
        if status != 0:
            print(f"ACADOS solver failed with status {status}")
        u0 = acados_ocp_solver.get(0, "u")
        
        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        simU[i, :] = u0

        simX[i+ 1, :] = xcurrent
        simX[i + 1, 2] = xcurrent[2]*cos(xcurrent[4])-xcurrent[3]*sin(xcurrent[4])
        simX[i + 1, 3] = xcurrent[2]*sin(xcurrent[4])+xcurrent[3]*cos(xcurrent[4])
        simX[i + 1, -2] = xcurrent[-2]+ arctan(xcurrent[3]/ xcurrent[2])# representation of theta 
        
        # update initial condition
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
        
        print('x' , xcurrent)
        print('y' , trajectory[:, i+1])
        print('u', simU[i,:])
    plot_trajectory(simX, simU, yref_)
    

def create_ocp_solver_BodyFrame() -> AcadosOcp:
    ocp = AcadosOcp()
    model = TimeDynBicyclePacejka()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([5*1e1, 5*1e0, 5*1e2, 5*1e-1, 5*1e1,5*1e2,1e-1,1e0]) #np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e2])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/(model.x[2]+1e-5)), model.x[5] , model.u)
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([5*1e2, 5*1e2, 1e1, 1e1, 5*1e1,1e1]) *dt_ocp
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

def closed_loop_simulationBodyFrame():
    ocp = create_ocp_solver_BodyFrame()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_dyn_pcb2.json')
    
    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    #sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_dyn_pcb2.json')
    
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
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))
    for stage in range(N_horizon+Nsim):
        yref_[stage, :] = trajectory[:, stage]
        
    for i in range(N):
    # update reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((trajectory[:, j+i],[0,0])))
        acados_ocp_solver.set(N_horizon, "yref", trajectory[:,i+N_horizon])
      
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

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["x", "y", "vx", "vy", "theta", "omega", "a", "delta"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsy, yref_[:N+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    closed_loop_simulationBodyFrame()

