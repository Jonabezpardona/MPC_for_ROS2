from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan,  tanh
from reference_trajectory_dynamic import *
from vehicle_params import  VehicleParams

'''
Taken models from acados examples and one thesis, 
Your track is too big for these dimensions of a car it needs to be more apropriate
BIG ATTENTION TO SAMPLING TIME AND OMEGA, IF THEY DO NOT CORRESPOND WELL IT WON'T WORK
THIS DEPENDS HEAVILY IN WHAT TRAJECTORY YOU ARE TRYING TO FOLLOW
'''

# ------------------ INITIAL CONFIGURATION ------------------
omega_ref = 0.005 # FOR SCURVE
dt_ocp = 0.005
dt_sim = 0.005


#omega_ref = 0.2 # for ellipse
#dt_ocp = 0.01
#dt_sim = 0.01

N_horizon = 20
Tf =N_horizon * dt_ocp


ellipse = EllipseTrajectory()
straight  = StraightLineTrajectory()
scurve = SCurveTrajectory()
trajectory,X0_ocp, T = scurve.time_ref(omega_ref, dt_ocp, N_horizon)
X0 = trajectory[:,0]

N = int(T /dt_ocp)
Nsim = int(T / dt_sim)
print("N",N)
print("Nsim",Nsim)

params = VehicleParams()
params.RaceCar43()


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


def create_ocp_solver() -> AcadosOcp:
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
    ocp.constraints.x0 = X0_ocp 
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

def closed_loop_simulation():
    ocp = create_ocp_solver()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_dyn_rc.json')
    
    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    #sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_dyn_rc.json')
    
    #simulation
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    yref_= np.zeros((Nsim+N_horizon,nx))
    xcurrent = X0_ocp
    simX[0, :] = xcurrent
    #simX[0, 2] = xcurrent[2]*cos(xcurrent[4])-xcurrent[3]*sin(xcurrent[4])
    #simX[0, 3] = xcurrent[2]*sin(xcurrent[4])+xcurrent[3]*cos(xcurrent[4])
    #simX[0, -2] = xcurrent[-2]+ arctan(xcurrent[3]/ (xcurrent[2]+1e-5))# representation of theta 
    
    print('x' , xcurrent)
    #print('X' , simX[0, :])
    print('y' , trajectory[:, 0])

    # initialize solver
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "x", X0_ocp)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))
    for stage in range(N_horizon+Nsim):
        yref_[stage, :] = trajectory[:, stage]
        
    for i in range(N):
    # update reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", trajectory[:, j+i])
        acados_ocp_solver.set(N_horizon, "yref", trajectory[:,i+N_horizon])
      
        status = acados_ocp_solver.solve()
        #u0 = acados_ocp_solver.solve_for_x0(xcurrent)
        if status != 0:
            print(f"ACADOS solver failed with status {status}")
        u0 = acados_ocp_solver.get(0, "u")
        simU[i, :] = u0
        
        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        simX[i+ 1, :] = xcurrent
        #simX[i + 1, 2] = xcurrent[2]*cos(xcurrent[4])-xcurrent[3]*sin(xcurrent[4])
        #simX[i + 1, 3] = xcurrent[2]*sin(xcurrent[4])+xcurrent[3]*cos(xcurrent[4])
        #simX[i + 1, -2] = xcurrent[-2]+ arctan(xcurrent[3]/ xcurrent[2])# representation of theta 
    
        # update initial condition
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
        
        print('u', simU[i,:])
        print('x' , xcurrent)
        print('X' , simX[i+ 1, :])
        print('y' , trajectory[:, i+1])
        
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
    closed_loop_simulation()
