'''
TIME KINEMATIC BICYCLE MODEL WITH DIFFERENT SAMPLING TIME FOR CONTROLLER AND SIMULATOR

SAMPLING TIMES OF CONTROLLER AND SIMULATOR ARE DIFFERENT
'''
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosSim, AcadosModel
from casadi import SX, vertcat, sin, cos, tan, arctan, fabs
from Modeling import * 

# ------------------ INITIAL CONFIGURATION ------------------


time_models = TimeDomainModels()
spatial_models = SpatialDomainModels()

# VEHICLE PARAMETERS  - HAVE TO BE THE SAME AS IN THE MODELING 
#params = VehicleParams()
#params.BoschCar() 

# ------------------ OCP SOLVERS ------------------

class TimeSolvers():
    def __init__(self, X0, N_horizon, Tf):
        self.X0 = X0
        self.N_horizon = N_horizon
        self.Tf = Tf 


    def Solver_NL_TimeKin(self, dt) -> AcadosOcp: #NONLINEAR SOLVER
        ocp = AcadosOcp()
        model = time_models.TimeKin()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx
        ocp.solver_options.N_horizon = self.N_horizon
        #Q_mat = 2 * np.diag([1e2,1e2, 1e3, 1e2])  #ellipse
        Q_mat = 2*np.diag([1e1, 1e1 , 1e1, 1e1])  #s_surve

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS" 
        ocp.cost.W = scipy.linalg.block_diag(Q_mat)
        ocp.cost.yref = np.zeros((nx,)) #ny
        ocp.model.cost_y_expr = vertcat( model.x[:3], model.x[3]+arctan(params.lr*tan(model.u[1])/params.L) )
        #terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag([1e1, 1e1 , 1e1])*dt
        #yref_e = np.array([10.0, 0.0, 0.0]) #ellipse path
        yref_e = np.array([50.0, 0.0, 0.0])  #s curve path
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[:3]) 
    
        # set constraints                                                                                                               
        ocp.constraints.lbu = np.array([ -3, -np.deg2rad(60)])
        ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
        ocp.solver_options.tf = self.Tf
        #ocp.solver_options.qp_solver_cond_N =   # MUST BE INTEGER, ONLY FOR PARTIAL CONDENSING
        return ocp

    def Solver_L_TimeKin(self,dt) -> AcadosOcp: # LINEAR SOLVER
        ocp = AcadosOcp()
        model = time_models.TimeKin()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx
        ocp.solver_options.N_horizon = self.N_horizon
        #Q_mat = 2 * np.diag([1e2,1e2, 1e3, 1e2])  #ellipse
        Q_mat =  np.diag([1e2, 1e2 , 1e3, 1e2])  #s_surve

        #path const
        ocp.cost.cost_type = "LINEAR_LS" 
        ocp.cost.W = scipy.linalg.block_diag(Q_mat)
        ocp.cost.yref = np.zeros((ny,)) 
        #terminal cost    Q_mat = 2 * np.diag([1e2, 1e2, 0.0, 0.0, 0.0,0.0])  

        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W_e =2 * np.diag([1e1, 1e1 , 1e1, 1e1])*dt
        #yref_e = np.array([10.0, 0.0, 0.0, np.pi/2]) #ellipse path
        yref_e = np.array([50.0, 0.0, 0.0, 0.0])  #s curve path
        ocp.cost.yref_e = yref_e
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[nx : nx + nu, 0:nu] = np.eye(nu)
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nx)

        # set constraints                                                                                                               
        ocp.constraints.lbu = np.array([ -3, -np.deg2rad(60)])
        ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0
        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
        ocp.solver_options.tf = self.Tf
        #ocp.solver_options.qp_solver_cond_N =   # MUST BE INTEGER, ONLY FOR PARTIAL CONDENSING
        return ocp

    def Solver_TimeDyn_Accx(self,dt) -> AcadosOcp: # it can be either AcadosOcp or AcadosSim
        ocp = AcadosOcp()
        model = time_models.TimeDyn_AccX()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx
        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat =  np.diag([5*1e2, 5*1e2, 1e1, 5*1e0, 1e1,1e0,1e-1,1e0]) #np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e2])  

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS" 
        ocp.cost.W = scipy.linalg.block_diag(Q_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/(model.x[2]+1e-5)), model.x[5] , model.u)
        #terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e =2 * np.diag([5*1e1, 5*1e0, 1e1, 1e1, 5*1e1,1e1]) *dt
        yref_e = np.array([10, 0.0, 0.0, 0.0, np.pi/2, 0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = model.x #vertcat(model.x[:4]) 

        # set constraints                                                                                                               
        ocp.constraints.lbu = np.array([-5,-np.deg2rad(45)])
        ocp.constraints.ubu = np.array([5, np.deg2rad(45)])
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.x0 = self.X0
        #ocp.constraints.lbx = np.array([-10, -10])
        #ocp.constraints.ubx = np.array([10, 10])
        #ocp.constraints.idxbx = np.array([2,3])

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = self.Tf
        return ocp
        
    def Solver_TimeDyn_LinVL(self,dt, a_max) -> AcadosOcp:
        ocp = AcadosOcp()
        model = time_models.TimeDyn_LinVL(a_max)
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx
        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = np.diag([1e1, 1e1, 1e1, 1e-1, 1e0, 1e-1, 1e-2])  # adjusted length to match nx=7
        R_mat = np.diag([1e-1, 1e-1])
        # path cost
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4] + arctan(model.x[3] / (model.x[2] + 1e-5)), model.x[5], model.x[6], model.u)

        # terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag([5*1e1, 1e1, 5*1e1, 1e0, 1e-1, 1e-1, 1e-2]) * dt
        yref_e = np.array([7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = model.x

        # set constraints
        ocp.constraints.lbu = np.array([-a_max, -np.deg2rad(30)])
        ocp.constraints.ubu = np.array([a_max, np.deg2rad(30)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = self.Tf
        return ocp

    def Solver_TimeDyn_RC(self,dt) -> AcadosOcp:
        ocp = AcadosOcp()
        model =time_models.TimeDyn_RC()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx
        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = 2 * np.diag([5*1e1, 5*1e2, 1e1, 1e0, 5*1e0,1e1])  

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS" 
        ocp.cost.W = Q_mat#scipy.linalg.block_diag(Q_mat)
        ocp.cost.yref = np.zeros((nx,))
        ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/(model.x[2]+1e-5)), model.x[5] )#model.x #vertcat(model.x[:2], model.x[2]*cos(model.x[4])-model.x[3]*sin(model.x[4]), model.x[2]*sin(model.x[4])+model.x[3]*cos(model.x[4]), model.x[4]+arctan(model.x[3]/ (model.x[2]+1e-5)), model.x[-1])
        #terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e =Q_mat * dt#2 * np.diag([5*1e1, 5*1e1, 1e0, 1e0, 5*1e1,1e1]) 
        yref_e = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = model.x #vertcat(model.x[:4]) 

        # set constraints                                                                                                               
        ocp.constraints.lbu = np.array([-5,-np.deg2rad(65)])
        ocp.constraints.ubu = np.array([5, np.deg2rad(65)])
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.x0 = self.X0
        #ocp.constraints.lbx = np.array([-10, -10])
        #ocp.constraints.ubx = np.array([10, 10])
        #ocp.constraints.idxbx = np.array([2,3])

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = self.Tf
        return ocp


class SpatialSolvers():

    def __init__(self,s_ref, kappa, X0, N_horizon, Tf):
        self.s_ref= s_ref
        self.kappa = kappa
        self.X0 = X0
        self.N_horizon = N_horizon
        self.Tf = Tf 

    def Solver_SpatialKin(self,ds) -> AcadosOcp: #NONLINEAR SOLVER
        ocp = AcadosOcp()
        model = spatial_models.SpatialKin(self.kappa)
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 2 + nu
        ny_e = 2

        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = np.diag([1e1,2*1e2])  
        R_mat =  np.diag([1e-1,1e-1])


        ocp.parameter_values  = np.array([self.s_ref[0]])

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(params.lr*tan(model.u[1])/params.L),model.x[1], model.u) #
    
        #terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = Q_mat *ds
        yref_e = np.array([0.0, 0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[:2]) 

        # set constraints  - this is for the input                                                                                                             
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(60)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(60)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

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
        ocp.solver_options.tf = self.Tf
        return ocp
        

    def Solver_SpatialDyn_Accx(self, ds) -> AcadosOcp:
        ocp = AcadosOcp()
        model = spatial_models.SpatialDyn_AccX(self.kappa)
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 3 + nu
        ny_e = 3

        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = np.diag([5*1e1,1e2,1e1])  
        R_mat =  np.diag([1e-1,1e-1])

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1],model.x[4], model.u) #
    
        #terminal costs
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        Q_mat = np.diag([1e1,5*1e1])
        ocp.cost.W_e = Q_mat*ds
        yref_e = np.array([0.0, 0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1]) 
    
        ocp.parameter_values  = np.array([self.s_ref[0]])
        # set constraints on the input                                                                                                             
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        # constraints on the states
        ocp.constraints.lbx = np.array([ -np.deg2rad(40), -0.5])
        ocp.constraints.ubx = np.array([ np.deg2rad(40), 0.5])
        ocp.constraints.idxbx = np.array([0,1])

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 70
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.tf = self.Tf
        return ocp

    def Solver_SpatialDyn_LinVL(self,ds, a_max) -> AcadosOcp:
        ocp = AcadosOcp()
        model = spatial_models.SpatialDyn_LinVL(self.kappa, a_max)
        ocp.model = model

        nx = model.x.rows()
        nu = model.u.rows()

        # keep same outputs as before (lateral/heading errors + vx + controls)
        ny = 3 + nu    # [epsi+beta, e_y, vx] + u
        ny_e = 2       # terminal: [epsi+atan(vy/vx), e_y]

        ocp.solver_options.N_horizon = self.N_horizon

        Q_mat = np.diag([2e1, 1e1, 1e0])   # weights for [epsi term, e_y, vx]
        R_mat = np.diag([1e-1, 1e-1])

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))

        # cost y expression - update indices for vx, vy
        # model.x[0] is e_psi ; model.x[5] is vy ; model.x[4] is vx
        ocp.model.cost_y_expr = vertcat(
            model.x[0] + arctan(model.x[5] / (model.x[4] + 1e-5)),
            model.x[1],
            model.x[4],
            model.u
        )

        # terminal
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        Qe = np.diag([1e1, 5e1])
        ocp.cost.W_e = Qe * ds
        ocp.cost.yref_e = np.array([0.0, 0.0])
        ocp.model.cost_y_expr_e = vertcat(
            model.x[0],
            model.x[1]
        )

        # parameter init (starting s)
        ocp.parameter_values = np.array([self.s_ref[0]])

        # input bounds
        ocp.constraints.lbu = np.array([-1.0, -np.deg2rad(30)])
        ocp.constraints.ubu = np.array([1.0, np.deg2rad(30)])
        ocp.constraints.idxbu = np.array([0, 1])

        # initial condition (note: X0 must have size nx)
        ocp.constraints.x0 = self.X0  # <-- X0 ensured to have length nx

        # simple bounds on epsi and e_y as before
        ocp.constraints.lbx = np.array([-np.deg2rad(80), -0.5])
        ocp.constraints.ubx = np.array([ np.deg2rad(80), 0.5])
        ocp.constraints.idxbx = np.array([0, 1])

        # solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 70
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.tf = self.Tf
        return ocp

    def Solver_SpatialDyn_RC(self,ds) -> AcadosOcp:
        ocp = AcadosOcp()
        model = spatial_models.SpatialDyn_RC(self.kappa)
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 2 + nu
        ny_e = 2

        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = np.diag([1e2,3*1e1])  
        R_mat =  np.diag([1e-1,1e-1])

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ model.x[4]+1e-4), model.x[1], model.u) #
    
        #terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = Q_mat*ds
        yref_e = np.array([0.0, 0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-4)), model.x[1]) 
    
        ocp.parameter_values  = np.array([self.s_ref[0]])
        # set constraints on the input                                                                                                             
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        # constraints on the states
        ocp.constraints.lbx = np.array([ -np.deg2rad(80), -0.85])
        ocp.constraints.ubx = np.array([ np.deg2rad(80), 0.85])
        ocp.constraints.idxbx = np.array([0,1])

        # set options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 70
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.tf = self.Tf
        return ocp



