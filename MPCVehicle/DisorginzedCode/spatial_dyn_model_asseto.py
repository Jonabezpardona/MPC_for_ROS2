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
ds = 0.05
omega = 0.3#0.08

# scurve
#ds = 0.03
#omega = 0.3

# N horizon values
N_horizon = 75
Tf =N_horizon * ds


ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()
straight = StraightLineTrajectory()
y_ref, S, s_ref, kappa_ref  = straight.spatial_ref(ds, N_horizon)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

X0 = y_ref[:,0]
N = int(S/ds)


def SpatialDynamicAsseto():
    #States
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    X = SX.sym('X')
    Y= SX.sym('Y')
    vx= SX.sym('vy')
    vy= SX.sym('vy')
    psi= SX.sym('psi')
    omega= SX.sym('omega')
    x = vertcat(e_psi, e_y,X,Y,vx,vy,psi,omega)

    s = SX.sym('s') # parameter

    # Inputs
    a= SX.sym('a')
    delta= SX.sym('delta')
    u = vertcat(a, delta)

    # ODE equations
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    X_dot = SX.sym('dX')
    Y_dot= SX.sym('dY')
    vx_dot= SX.sym('dvy')
    vy_dot= SX.sym('dvy')
    psi_dot= SX.sym('dpsi')
    omega_dot= SX.sym('domega')

    xdot = vertcat(e_psi_dot, e_y_dot,  X_dot,Y_dot,vx_dot,vy_dot,psi_dot,omega_dot)


    vx_frontwheel = vx*cos(delta) + (vy + omega*lf)*sin(delta)
    vy_frontwheel = (vy + omega*lf)*cos(delta) - vx*sin(delta)
        
    # lateral slip angles 1e-5
    alpha_f = arctan(vy_frontwheel/(vx_frontwheel+1e-5))
    alpha_r = arctan((vy - omega*lr)/ (vx+1e-5))

    # longitudial force acting only on rear wheel 
    Fx = ((Tm0 + Tm1 * vx) * a)#- (Tr0*(1-tanh(Tr1*vx)) + Tr2 * vx**2)

    # lateral forces acting on the tires
    Fcf = Df * sin(Cf*arctan(Bf*alpha_f))
    Fcr = Dr * sin(Cr*arctan(Br*alpha_r))

    sdot = (vx* cos(e_psi) - vy *sin(e_psi))/(1 - kappa(s)* e_y)
    dX = (vx*cos(psi)- vy*sin(psi))/(sdot+1e-5)
    dY = (vx*sin(psi) + vy* cos(psi))/(sdot+1e-5)
    dvx = (vy*omega + 1/m * (Fx + Fcf*sin(delta)))/(sdot)
    dvy = (- vx*omega + 1/m * (Fcf*cos(delta) + Fcr))/(sdot+1e-5)
    dpsi = (omega)/(sdot+1e-5)
    domega = (1/Iz*(lf*Fcf*cos(delta) - lr*Fcr))/(sdot+1e-5)
    de_psi = (dpsi)/(sdot+1e-5) - kappa(s)
    de_y = (vx  * sin(e_psi) + vy * cos(e_psi)) / (sdot+1e-5)
    

    p = vertcat(s)

    f_expl = vertcat(de_psi,de_y,dX, dY, dvx, dvy, dpsi, domega)
    f_impl = xdot - f_expl
    
    model = AcadosModel()
    model.name = 'SpatialDynBicycleAsseto'
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p 

    sdot_out = SX.sym('sdot_out')
    h = vertcat(sdot_out)  # Output vector
    model.h = h
    model.h_expression = sdot 
    return model

def CreateSolver_DynAsseto() -> AcadosOcp:
    ocp = AcadosOcp()
    model = SpatialDynamicAsseto()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu -3
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    #  ellipse
    Q_mat = np.diag([8*1e0, 8*1e1, 2*1e1, 2*1e0, 5*1e0]) #([8*1e1, 8*1e1, 2*1e0, 2*1e0, 5*1e0,1e1]) # works welll do not fcking touch 
    R_mat = np.diag([1e0, 5*1e0])
    
    #scurve
    #Q_mat = np.diag([5*1e2, 1e1, 1e-1, 1e-1, 1e-1,1e2]) 
    #R_mat = np.diag([1e-1, 1e-1])
    # path/stage cost
    ocp.cost.W = scipy.linalg.block_diag(Q_mat,R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(model.x[5]/ (model.x[4]+1e-5)),model.x[1], model.x[4], model.x[5], model.x[-1], model.u) 
  
    # terminal cost
    ocp.cost.W_e = Q_mat*ds
    ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.model.cost_y_expr_e = vertcat(model.x[0]+arctan(model.x[5]/ (model.x[4]+1e-5)),model.x[1], model.x[4], model.x[5], model.x[-1]) 
    
    ocp.parameter_values  = np.array([s_ref[0]])
    # constraints
    ocp.constraints.x0 = X0
    ocp.constraints.lbu =np.array([-a_max,-delta_max])
    ocp.constraints.ubu = np.array([a_max,delta_max])
    ocp.constraints.idxbu = np.array([0,1])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_iter_max = 200 

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
    sim.solver_options.T = ds
    sim.parameter_values = np.array([s_ref[0]])
    acados_sim_solver = AcadosSimSolver(sim)

    #inizialization of values
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((N+1,nx))
    simU = np.zeros((N,nu))
    predX = np.zeros((N +N_horizon+ 1, nx))
    predU = np.zeros((N+N_horizon, nu))

    xcurrent = X0
    simX[0,:] = X0
    predX[0,:] = X0

    #sovler inizialization
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "x", xcurrent)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[stage]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))

    # under the assumption that the control and the simulation have the same stepsize  
    for i in range(N):
        #updating the reference
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[:2,i+j],y_ref[4:6,i+j],[y_ref[-1,i+j]],[0.0,0.0])))
            acados_ocp_solver.set(j, "p", np.array([s_ref[i + j]]))
        acados_ocp_solver.set(N_horizon, "yref", np.concatenate((y_ref[:2,i+N_horizon],y_ref[4:6,i+N_horizon],[y_ref[-1,i+N_horizon]])))
        acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[i + N_horizon]]))

        status = acados_ocp_solver.solve()
        stats = acados_ocp_solver.print_statistics()
        #print(stats['qp_stat'], stats['qp_iter'])
        
        u0 = acados_ocp_solver.solve_for_x0(xcurrent)
        if status != 0:
            print(f"ACADOS solver failed with status {status}")

        
        u0 = acados_ocp_solver.get(0,"u")
        acados_sim_solver.set("p", np.array([ s_ref[i] ]))
        simU[i,:] = u0

        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        simX[i+1,:] = xcurrent
        simX[i + 1 , 0] = simX[i + 1 , 0]+ arctan(simX[i+1,5]/(simX[i+1,4]+1e-5)) # representation of epsi 
        simX[i + 1, -2] = simX[i + 1, -2] + arctan(simX[i+1,5]/(simX[i+1,4]+1e-5)) # representation of theta 



        #set constraints for the next control iteration
        acados_ocp_solver.set(0,"lbx",xcurrent)
        acados_ocp_solver.set(0,"ubx",xcurrent)

        #prints
        print('SREF' , s_ref[i])
        print('x', xcurrent)
        print('y',y_ref[:,i+1])
        print('u',u0)

    #plots
    plot_trajectory_in_space(simX,y_ref)
    plot_states(simX,simU,y_ref)


def plot_trajectory_in_space(simX,yref):
    plt.figure()
    plt.plot(simX[:,2],simX[:,3], label='Simulation')
    plt.plot(yref[2, :N+1], yref[3, :N+1], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

def plot_states(simX,simU,y_ref):
    timestampsx = np.linspace(0,(N+1)*ds,N+1)
    timestampsu = np.linspace(0,(N)*ds,N)

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = [ "epsi", "ey","x", "y", "vx", "vy", "theta", "omega", "a", "delta"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsx, y_ref[i,:N+1], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    CLSimulation_DynAsseto()
    
    