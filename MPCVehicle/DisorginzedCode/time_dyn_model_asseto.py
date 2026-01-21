from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import casadi
import matplotlib.pyplot as plt
import scipy.linalg
from reference_trajectory_dynamic import *

# ------- Vehicle Params --------
m = 1000.24
lf =  1.37
lr  = 1.35
Iz =  2997.48

Bf =  18.74
Cf = -1.41
Df  = 6580.13
Br  = 29.97
Cr  = -1.06
Dr  = 7152.6

Tm0 =  6531.97
Tm1 =  68.065
Tr0 = 726.99
Tr1 = -0.0495
Tr2 =  1.906

delta_max =  0.261
a_max = 3.0

# ------- Reference Generation --------
# ellipse
# ovo sad ne radi
dt = 0.1
omega = 0.1

# scurve
#dt = 0.03
#omega = 0.3

# N horizon values
N_horizon = 75
Tf =N_horizon * dt


ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()
straight = StraightLineTrajectory()
y_ref, X0, T = ellipse.time_ref(omega,dt, N_horizon) # reference is computed for body frame here
N = int(T/dt)

def TimeDynamicAsseto():
    #States
    X = SX.sym('X')
    Y= SX.sym('Y')
    vx= SX.sym('vy')
    vy= SX.sym('vy')
    psi= SX.sym('psi')
    omega= SX.sym('omega')
    x = vertcat(X,Y,vx,vy,psi,omega)

    # Inputs
    a= SX.sym('a')
    delta= SX.sym('delta')
    u = vertcat(a, delta)

    # ODE equations
    X_dot = SX.sym('dX')
    Y_dot= SX.sym('dY')
    vx_dot= SX.sym('dvy')
    vy_dot= SX.sym('dvy')
    psi_dot= SX.sym('dpsi')
    omega_dot= SX.sym('domega')
    xdot = vertcat(X_dot,Y_dot,vx_dot,vy_dot,psi_dot,omega_dot)


    vx_frontwheel = vx*cos(delta) + (vy + omega*lf)*sin(delta)
    vy_frontwheel = (vy + omega*lf)*cos(delta) - vx*sin(delta)
        
    # lateral slip angles 1e-5
    alpha_f = arctan(vy_frontwheel/(vx_frontwheel+1e-5))
    alpha_r = arctan((vy - omega*lr)/ (vx+1e-5))

    # longitudial force acting only on rear wheel 
    Fx = ((Tm0 + Tm1 * vx) * a)- (Tr0*(1-tanh(Tr1*vx)) + Tr2 * vx**2)

    # lateral forces acting on the tires
    Fcf = Df * sin(Cf*arctan(Bf*alpha_f))
    Fcr = Dr * sin(Cr*arctan(Br*alpha_r))

    dX = vx*cos(psi)- vy*sin(psi)
    dY = vx*sin(psi) + vy* cos(psi)
    dvx = vy*omega + 1/m * (-Fx + Fcf*sin(delta))
    dvy =  - vx*omega + 1/m * (Fcf*cos(delta) + Fcr)
    dpsi = omega
    domega = 1/Iz*(lf*Fcf*cos(delta) - lr*Fcr)

    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega)
    f_impl = xdot - f_expl
    
    model = AcadosModel()
    model.name = 'TimeDynBicycleAsseto'
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    return model


def f(x, u):
    #States
    X = x[0]
    Y = x[1]
    vx = x[2]
    vy = x[3]
    psi = x[4]
    omega = x[5]

    # Inputs
    a = u[0]
    delta = u[1]

    # ODE equations
    vx_frontwheel = vx*cos(delta) + (vy + omega*lf)*sin(delta)
    vy_frontwheel = (vy + omega*lf)*cos(delta) - vx*sin(delta)
        
    # lateral slip angles 1e-5
    alpha_f = np.arctan(vy_frontwheel/(vx_frontwheel+1e-5))
    alpha_r = np.arctan((vy - omega*lr)/ (vx+1e-5))

    # longitudial force acting only on rear wheel 
    Fx = ((Tm0 + Tm1 * vx) * a)- (Tr0*(1-np.tanh(Tr1*vx)) + Tr2 * vx**2)

    # lateral forces acting on the tires
    Fcf = Df * np.sin(Cf*np.arctan(Bf*alpha_f))
    Fcr = Dr * np.sin(Cr*np.arctan(Br*alpha_r))

    dX = vx*np.cos(psi)- vy*np.sin(psi)
    dY = vx*np.sin(psi) + vy*np.cos(psi)
    dvx = vy*omega + 1/m * (-Fx + Fcf*np.sin(delta))
    dvy =  - vx*omega + 1/m * (Fcf*np.cos(delta) + Fcr)
    dpsi = omega
    print(omega)
    domega = 1/Iz*(lf*Fcf*np.cos(delta) - lr*Fcr)

    return np.hstack([dX, dY, dvx, dvy, dpsi, domega])


def CreateSolver_DynAsseto() -> AcadosOcp:
    ocp = AcadosOcp()
    model = TimeDynamicAsseto()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    #  ellipse
    Q_mat = np.diag([8*1e1, 8*1e1, 2*1e0, 2*1e0, 5*1e0,1e1]) #([8*1e1, 8*1e1, 2*1e0, 2*1e0, 5*1e0,1e1]) # works welll do not fcking touch 
    R_mat = np.diag([1e0, 5*1e0])
    
    #scurve
    #Q_mat = np.diag([1e2, 1e2, 1e-1, 1e-3, 1e-1,1e-3]) 
    #R_mat = np.diag([1e-1, 1e-1])
    # path/stage cost
    ocp.cost.W = scipy.linalg.block_diag(Q_mat,R_mat)
    ocp.cost.yref = np.zeros((ny,))
    #ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/ (model.x[2]+1e-5)), model.x[-1], model.u)
    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    
    # terminal cost
    ocp.cost.W_e = Q_mat*dt
    ocp.cost.yref_e = np.array([50, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.model.cost_y_expr_e = model.x #vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/ (model.x[2]+1e-5)), model.x[-1])
    

    # constraints
    ocp.constraints.x0 = X0
    ocp.constraints.lbu =np.array([-a_max,-delta_max])
    ocp.constraints.ubu = np.array([a_max,delta_max])
    ocp.constraints.idxbu = np.array([0,1])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf

    return ocp 


def CLSimulation_DynAsseto():
    # OCP Model 
    ocp  = CreateSolver_DynAsseto()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp)

    #Simulation model - here the same as the OCP 
    sim = AcadosSim()
    sim.model = ocp.model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.T = dt
    acados_sim_solver = AcadosSimSolver(sim)

    #inizialization of values
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((N+1,nx))
    simU = np.zeros((N,nu))

    xcurrent = X0
    simX[0,:] = X0

    #sovler inizialization
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))

    # under the assumption that the control and the simulation have the same stepsize  
    for i in range(N):
        #updating the reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[:,i+j],[0.0,0.0])))
        #acados_ocp_solver.set(N_horizon,"yref",y_ref[:,i+N_horizon])
        acados_ocp_solver.set("yref_e",y_ref[:,i+N_horizon])

        status = acados_ocp_solver.solve()
        if status != 0:
            print(f"ACADOS solver failed with status {status}")
        
        u0 = acados_ocp_solver.get(0,"u")
        simU[i,:] = u0

        #xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        #xcurrent = rk4(f, xcurrent, u0, dt)
        simX[i+1,:] = xcurrent
        simX[i+1,-2] = simX[i+1,-2] + arctan(simX[i+1,3]/(simX[i+1,2]+1e-5))

        #set constraints for the next control iteration
        acados_ocp_solver.set(0,"lbx",xcurrent)
        acados_ocp_solver.set(0,"ubx",xcurrent)

        #prints
        print('x', xcurrent)
        print('y',y_ref[:,i+1])
        print('u',u0)

    #plots
    plot_trajectory_in_space(simX,y_ref)
    plot_states(simX,simU,y_ref)


def plot_trajectory_in_space(simX,yref):
    plt.figure()
    plt.plot(simX[:,0],simX[:,1], label='Simulation')
    plt.plot(yref[0, :N+1], yref[1, :N+1], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

def plot_states(simX,simU,y_ref):
    timestampsx = np.linspace(0,(N+1)*dt,N+1)
    timestampsu = np.linspace(0,(N)*dt,N)

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["x", "y", "vx", "vy", "theta", "omega", "a", "delta"]
    #omega_ = np.gradient(simX[:, 4],dt)
    for i in range(nx):
        #ax[i].plot(timestampsx, simX[:, i] if i != 5 else omega_)
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsx, y_ref[i,:N+1], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


def rk4(f, x, u, dt):
    k1 = dt * f(x, u)
    k2 = dt * f(x+k1/2, u)
    k3 = dt * f(x+k2/2, u)
    k4 = dt * f(x+k3,u)
    xnew = x + (k1+2*k2+2*k3+k4)/6
    return xnew


def euler(f, x, u, dt):
    return x + dt * f(x, u)



if __name__ == "__main__":
    CLSimulation_DynAsseto()
    
    