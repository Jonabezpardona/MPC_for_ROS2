from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant
from reference_trajectory_dynamic import *
from ref_time2spatial import *
from vehicle_params import VehicleParams

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.001
ds_sim = 0.001
N_horizon =50
Tf =N_horizon *ds_ocp

straight_line = StraightLineTrajectory()
ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

y_ref, S, s_ref, kappa_ref= scurve.spatial_ref(ds_ocp, N_horizon)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)
N = int(S/ds_ocp) 
Nsim = int(S/ds_sim)
X0 = y_ref[0,:]
print(X0)

params = VehicleParams()
params.RaceCar43()  

# ------------------ MODELING ------------------
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
    Fx_d = (params.Cm1 -params.Cm2*vx)*D - params.cr2*vx**2 - params.cr0 * tanh(params.cr3 *vx)

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
def CreateOcpSolver_SpatialKin() -> AcadosOcp:
    ocp = AcadosOcp()
    model = SpatialDynModel_RC()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 2 + nu
    ny_e = 2

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([3e1,3*1e1])  
    R_mat =  np.diag([1e-1,1e-1])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1], model.u) #
  
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
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
    ocp.constraints.lbx = np.array([ -np.deg2rad(40), -0.5])
    ocp.constraints.ubx = np.array([ np.deg2rad(40), 0.5])
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


# ------------------CLOSED LOOP ------------------
def ClosedLoopSimulation():
    #AcadosOcpSovler
    ocp = CreateOcpSolver_SpatialKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_RCspatial.json')

    #AcadosIntegrator
    sim = AcadosSim()
    sim.model = ocp.model   
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = ds_sim # simulation step size [s]
    sim.parameter_values = np.array([s_ref[0]])
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_RCspatial.json')

    #simulation
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    Beta = np.zeros((Nsim, 1))
    predX = np.zeros((Nsim +N_horizon+ 1, nx))
    predU = np.zeros((Nsim+N_horizon, nu))

    #initialization
    xcurrent = X0
    print('x0', X0)
    predX[0,:] = xcurrent
    simX[0, :] = xcurrent
    s_prev = - ds_sim
    s_sim = s_ref[0]
    k = 0

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[stage]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0])) #warm start

    for i in range(N): 
        # Reference for horizon
        if (s_sim- s_prev >= ds_ocp) or(s_sim > s_ref[k+1]):
            for j in range(N_horizon):
                acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[j+k, :2],[0.0, 0.05])))
                acados_ocp_solver.set(j, "p", np.array([s_ref[k + j]]))
            acados_ocp_solver.set(N_horizon, "yref", y_ref[ k+N_horizon,:2])
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
        acados_sim_solver.set("p", np.array([ s_ref[k+1]]))
        xprev = xcurrent
        xcurrent = acados_sim_solver.simulate(xcurrent, u0, s_sim) 
        simU[i, :] = u0
        simX[i + 1, :] = xcurrent
        simX[i + 1 , 0] = simX[i + 1 , 0]+ np.arctan(simX[i + 1 , 5]/ (simX[i + 1 , 4])) #arctan (simX[i + 1 , 0]) # representation of epsi, it only affects what we are seeing on the graph, not on interation itself.
        simX[i + 1, -2] = simX[i + 1, -2] + np.arctan(simX[i + 1 , 5]/ (simX[i + 1 , 4])) #arctan () # representation of theta 
        #s_sim,_ ,_= time2spatial(xcurrent[2], xcurrent[3], xcurrent[6],s_ref,y_ref[:,2:5])
        s_sim = s_sim + np.sqrt((xcurrent[2]-xprev[2])**2+(xcurrent[3]-xprev[3])**2)
        acados_ocp_solver.set(0, "lbx",xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        # prints
        print('SREF' , s_ref[k])
        print('Ssim' , s_sim)
        print('x' , xcurrent)
        print('y',y_ref[k+1,:])
        print('u', simU[i,:])

    plot_trajectory(simX, simU, y_ref)
    

def plot_trajectory(simX, simU, y_ref):
    # Time vectors
    timestampsx = np.linspace(0, (Nsim+1)*ds_sim, Nsim+1)
    timestampsu = np.linspace(0, Nsim*ds_sim, Nsim)
    timestampsy = np.linspace(0, (N+1)*ds_ocp, N+1)

    # --- Trajectory Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(simX[:,2], simX[:,3], label=r'Simulation', linewidth=2)
    plt.plot(y_ref[:N+1,2], y_ref[:N+1,3], '--', alpha=0.9, label=r'Reference', color="C1", linewidth=1.5)
    plt.xlabel(r'$x [m]$')
    plt.ylabel(r'$y[m]$')
    plt.axis('equal')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # --- 1. Errors ---
    fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    error_labels = [r'$e_{\psi}[rad]$', r'$e_y$[m]']
    for i in range(2):
        ax1[i].plot(timestampsx, simX[:, i],label = r'Simulation',  color="#808000", linewidth=2 )
        ax1[i].plot(timestampsy, y_ref[:N+1,i], '--', label = r'Reference', color="C1", linewidth=1.5)
        ax1[i].set_ylabel(error_labels[i])
        ax1[i].grid(True, linestyle="--", alpha=0.4)
    ax1[-1].set_xlabel(r'Arc length[m]')
    ax1[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
    ax1[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
    ax1[0].set_title("Lateral Tracking Error", fontsize=12, fontweight='bold')
    ax1[1].set_title("Heading Tracking Error", fontsize=12, fontweight='bold')
    ax1[-1].legend(fontsize=11, loc='lower right')
    fig1.tight_layout(rect=[0, 0.05, 1, 1])

    # --- 2. States ---
    state_indices = [2, 3, 4, 5, 6, 7]  # x, y, vx, vy, theta, omega
    state_labels = [r'$x[m]$', r'$y[m]$', r'$v_x[\frac{m}{s}]$', r'$v_y[\frac{m}{s}]$', r'$\theta[rad]$', r'$\omega[\frac{rad}{s}]$']
    fig2, ax2 = plt.subplots(len(state_indices), 1, sharex=True, figsize=(8, 14))
    for j, idx in enumerate(state_indices):
        ax2[j].plot(timestampsx, simX[:, idx],label = r'Simulation',  color="C0", linewidth=2)
        ax2[j].plot(timestampsy, y_ref[:N+1, idx], '--', label = r'Reference',   color="#ffd166", linewidth=1.5)
        ax2[j].set_ylabel(state_labels[j])
        ax2[j].grid(True, linestyle="--", alpha=0.4)
    ax2[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax2[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax2[4].set_title("Direction Angle", fontsize=12, fontweight='bold')
    ax2[-1].set_xlabel(r'Arc length$[m]$')
    ax2[-1].legend(fontsize=11, loc='upper right')
    ax2[-1].grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    

    # --- 3. Controls ---
    control_labels = [r'$D$', r'$\delta[rad]$']

    for i in range (simU.shape[0]):
        if simU[i, 0] < 0:
            simU[i, 0] = 0 

    fig3, ax3 = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax3[0].plot(timestampsu, simU[:, 0],color="C2", linewidth=2)
    ax3[0].set_ylabel(control_labels[0])
    ax3[0].set_title("Duty Cycle of the motor", fontsize=12, fontweight='bold')
    ax3[0].grid(True, linestyle="--", alpha=0.4)
    

    ax3[1].plot(timestampsu, simU[:, 1],color="C3", linewidth=2)
    ax3[1].set_ylabel(control_labels[1])
    ax3[1].set_xlabel(r'Arc length$[m]$')
    ax3[1].set_title("Steering Angle", fontsize=12, fontweight='bold')
    ax3[1].grid(True, linestyle="--", alpha=0.4)
    fig3.tight_layout(rect=[0, 0.05, 1, 1])

    plt.show()


if __name__ == "__main__":
    ClosedLoopSimulation()
