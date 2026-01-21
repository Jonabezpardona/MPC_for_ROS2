'''
The bicycle model reduces interactions of four wheels into a two-wheel system. Modeling a car as if it
had one front and one rear wheel helps to more easily analyze some of the aspects
of vehicle behavior, stability, control, and steering. Both longitudinal and lateral
position are captured, maintaining accuracy without significant computational
demand.


Kinematic Bicycle Model describes the motion without considering the dynamics and 
is mostly used for the lateral vehicle control. The most important assumption of 
this model is that all slip angles on the wheels are equal to zero.

Dynamic Bicycle Model is more complex, which can be both good and bad. 
In terms of accuracy, this will make our modeling more precise, thus 
improving tracking capabilities, but will make control more computationally expensive. 
We use the Second Newton Law to form the set of equations that describe the model. 
Compared to the kinematic bicycle model, we are not only describing profiles of desired 
global position and angle, but also their respective derivatives - linear and angular velocities.



Modeling of the vehicle in Spatial Domain 

Spatial reformulation requires rewriting of the systems dynamics using spatial
derivative instead of time derivative. Meaning that, instead of describing our
states with respect to time, we describe it with respect to the arc of length along
the track. The reference trajectory is then parameterized in s and the curvature
of the trajectory k = 1/ρ , where ρ is the instantaneous curvature radius, is locally
defined as: 
    k = (x'y'' - y'x'')/((x'2 + y'2)^3/2)

where f' = df(s)/ds and f'' = d2f(s)/ds^2
We introduce two additional states - angular and lateral error. They can be computed as:
eψ = ψ - ψs
ey = (Ys - Y ) cos(ψs) + (Xs - X) sin(ψs)


The state vector ξ is differentiated w.r.t. s using the chain rule as:
ξ′ = dξ/ds = dξ/dt · dt/ds = dξ/dt · 1/ṡ = ξ̇ /ṡ

'''
from acados_template import AcadosModel
import scipy.linalg
from casadi import SX, vertcat, sin, cos, tan, tanh, arctan, fabs
from vehicle_params import VehicleParams

 
# VEHICLE PARAMETERS 
params = VehicleParams()
params.BoschCar() 



# ------------------ MODELING ------------------
class TimeDomainModels(): 
    # KINEMATIC MODEL 
    def TimeKin(self):
        model_name = "TimeKin"
        # STATES AND CONTROL
        # SX is scalar symbolic type
        x = SX.sym("x")
        y = SX.sym("y")
        v = SX.sym("v")
        psi = SX.sym("psi")
        x = vertcat(x, y, v, psi) # vertically concatenate


        a = SX.sym("a")
        delta = SX.sym("delta")
        u = vertcat(a, delta)

        
        # xdot
        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        v_dot = SX.sym("v_dot")
        psi_dot = SX.sym("psi_dot")
        xdot = vertcat(x_dot, y_dot, v_dot, psi_dot)


        # dynamics
        beta = arctan(params.lr * tan(delta) / params.L) # slipping angle
        f_expl = vertcat(v * cos(psi+beta), v * sin(psi+beta), a ,v *sin(beta) /params.lr) 
        
        
        f_impl = xdot - f_expl
        model = AcadosModel()
        model.name = model_name
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u

        model.t_label = "$t$ [s]"
        model.x_labels = ["$x$", "$y$", "$v$", "$\\psi$"]
        model.u_labels = [ "$a$", "$delta$"]

        return model

    # LATERAL FORCFES AS PACEJKA MODEL, INSTEAD OF LONGITUDIAL FORCES LONGITUDAL ACCELERATION IS USED
    # assume that all the acceleration is applied to x (linear acceleration)
    def TimeDyn_AccX(self):
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
        beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta
        beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))
        
        Fx_d = 1/2*params.ro* params.Cz* params.Az* vx*fabs(vx)
        # Lateral tire forces - simplified Pacejka # forget about them, try 
        Fc_f = -params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
        Fc_r = -params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r) )

        Fyf = Fc_f*cos(delta) 
        Fyr = Fc_r


        # DYNAMICS
        dX   = vx*cos(psi) - vy*sin(psi)
        dY  = vx*sin(psi) + vy*cos(psi)
        dvx   = vy*omega + a  +  (-Fx_d - Fc_f*sin(delta))/params.m # we consider that this is completly longitudial acceleration
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

    # VIRTUAL DRIVER FOR HIGH PERFORMANCE
    # has additional state - the change of the vertical load 
    def TimeDyn_LinVL(self, a_max): 
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
        # NEEDS NORMALIZED ACC/BREAKING COEFF !!!!!
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
        vy_r = vy - params.lr * omega  
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
        Fc_f = -params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )  
        Fc_r = -params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r) )

        Fzf_0 = params.m * params.g * params.lr / (params.L) 
        Fzr_0 = params.m * params.g * params.lf / (params.L) 
        # this is if my calculations are correct

        Fzf  = Fzf_0 - delta_Fz
        Fzr  = Fzr_0 + delta_Fz

        # Longitudial tire forces - linearly dependent of the vertical load
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

    # Race Car Model scaled 1/43, only works on that size of the model, since  needed motor parameters are unknown
    def TimeDyn_RC(self):
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


class SpatialDomainModels():

    # KINEMATIC MODEL 
    def SpatialKin(self,kappa):
        model_name = "SpatialKinematicBicycle_model"

        ## CasADi Model
        s = SX.sym('s')
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        v= SX.sym("v")
        e_psi = SX.sym('e_psi')
        e_y = SX.sym('e_y')
        x = vertcat(e_psi, e_y, x, y, v, psi)

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
        xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, v_dot, psi_dot)

        beta = arctan(params.lr * tan(delta) / params.L) # slipping angle
        vx = v* cos(psi+beta)
        vy = v* sin(psi+beta)
        dpsi = v *sin(beta) / params.lr
        
        #Spatial dynamics dx/ds = f(x,u)
        sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta) *sin(e_psi))/(1 - kappa(s)* e_y)
        dx_ds    = vx / (sdot)
        dy_ds    = vy / (sdot)
        dv_ds    = a / (sdot)
        dpsi_ds  = (dpsi) / (sdot)
        d_e_psi = (dpsi)/(sdot) - kappa(s)
        d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) / (sdot)
        f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dv_ds , dpsi_ds)
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
        model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$v$", "$\\psi$"]
        model.u_labels = [ "$a$", "$delta$"]
        model.p_labels    = ["$s$"] 
        model.name = model_name
        return model

    # LATERAL FORCFES AS PACEJKA MODEL, INSTEAD OF LONGITUDIAL FORCES LONGITUDAL ACCELERATION IS USED
    # assume that all the acceleration is applied to x (linear acceleration)
    def SpatialDyn_AccX(self,kappa):
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
        beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta 
        beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))
        
        Fx_d = 1/2*params.ro* params.Cz* params.Az* vx*fabs(vx)
        # Lateral tire forces - simplified Pacejka # forget about them, try 
        Fc_f = -params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
        Fc_r = - params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r) )

        Fyf = Fc_f*cos(delta) 
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

    # VIRTUAL DRIVER FOR HIGH PERFORMANCE
    # has additional state - the change of the vertical load 
    def SpatialDyn_LinVL(self,kappa, a_max):
        """
        Spatial-domain dynamic double-bicycle model with delta_Fz (normal load transfer)
        State order: e_psi, e_y, x, y, vx, vy, psi, omega, delta_Fz
        Controls: a, delta
        Parameter: s (arc length) used for kappa(s) interpolant
        """
        model_name = "SpatialDynamicBicycle_deltaFz"

        # symbolic states
        s = SX.sym('s')
        e_psi = SX.sym('e_psi')
        e_y   = SX.sym('e_y')
        x_pos = SX.sym('x')
        y_pos = SX.sym('y')
        vx    = SX.sym('vx')
        vy    = SX.sym('vy')
        psi   = SX.sym('psi')
        omega = SX.sym('r')
        delta_Fz = SX.sym('delta_Fz')

        x = vertcat(e_psi, e_y, x_pos, y_pos, vx, vy, psi, omega, delta_Fz)

        # controls
        a = SX.sym('a')
        delta = SX.sym('delta')
        u = vertcat(a, delta)

        # xdot (time derivatives) as symbolic placeholders - casadi requires xdot vector for implicit model
        e_psi_dot = SX.sym('e_psi_dot')
        e_y_dot   = SX.sym('e_y_dot')
        x_dot     = SX.sym('x_dot')
        y_dot     = SX.sym('y_dot')
        vx_dot    = SX.sym('vx_dot')
        vy_dot    = SX.sym('vy_dot')
        psi_dot   = SX.sym('psi_dot')
        omega_dot = SX.sym('omega_dot')
        delta_Fz_dot = SX.sym('delta_Fz_dot')
        xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, vx_dot, vy_dot, psi_dot, omega_dot, delta_Fz_dot)

        # short names to params
        lf = params.lf
        lr = params.lr

        # vehicle kinematics (wheelframe velocities)
        vy_f = vy + lf * omega
        vy_r = vy - lr * omega
        vx_f = vx
        vx_r = vx

        # velocities in wheel frames (front accounts for steering)
        vc_f = vy_f * cos(delta) - vx_f * sin(delta)   # lateral component front in wheel frame
        vc_r = vy_r
        vl_f = vy_f * sin(delta) + vx_f * cos(delta)   # longitudinal component front in wheel frame
        vl_r = vx_r

        # slip angles
        beta_f = arctan(vc_f/(vl_f + 1e-5))
        beta_r = arctan(vc_r/(vl_r + 1e-5))

        # aero drag
        Fx_d = 0.5 * params.ro * params.Cz * params.Az * vx * fabs(vx)

        # lateral tire forces (simplified Pacejka)
        Fc_f = - params.Dcf * sin( params.Ccf * arctan( params.Bcf * beta_f) )
        Fc_r = - params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r ) )

        # static normal loads (nominal)
        Fzf_0 = params.m * params.g * lr / (params.L)
        Fzr_0 = params.m * params.g * lf / (params.L)

        # include delta_Fz effect on normal loads
        Fzf = Fzf_0 - delta_Fz
        Fzr = Fzr_0 + delta_Fz

        # longitudinal traction/braking: saturate via tanh(a/a_max) like your time-model
        gamma = tanh(a / a_max)
        # Use current normal loads (Fzf, Fzr) to compute max longitudinal force (more physically consistent)
        Fl_f = gamma * params.mi * Fzf
        Fl_r = gamma * params.mi * Fzr

        # wheel forces (combine longitudinal and lateral, account for steering at front)
        Fx_f = Fl_f * cos(delta) - Fc_f * sin(delta)
        Fx_r = Fl_r
        Fx = Fx_f + Fx_r

        Fy_f = Fl_f * sin(delta) + Fc_f * cos(delta)
        Fy_r = Fc_r

        # pitch/roll induced vertical transfer (same algebraic formula as your time model)
        deltaz = - params.h * Fx / (2 * params.L)
        tau_z = 0.5
        ddelta_Fz = (deltaz - delta_Fz) / tau_z

        # time-domain dynamics (as in your time model)
        dX   = vx * cos(psi) - vy * sin(psi)
        dY   = vx * sin(psi) + vy * cos(psi)
        dvx  = vy * omega + (Fx - Fx_d) / params.m
        dvy  = - vx * omega + (Fy_f + Fy_r) / params.m
        dpsi = omega
        domega = (lf * Fy_f - lr * Fy_r) / params.I_z

        # spatial rate sdot (arc-length derivative)
        # uses curvature kappa(s) via interpolant kappa(s) available in your workspace
        sdot = (vx * cos(e_psi) - vy * sin(e_psi)) / (1 - kappa(s) * e_y + 1e-8)

        # convert time derivatives to spatial derivatives dx/ds = (dx/dt) / sdot
        dx_ds   = dX   / (sdot)
        dy_ds   = dY   / (sdot)
        dvx_ds  = dvx  / (sdot)
        dvy_ds  = dvy  / (sdot)
        dpsi_ds = dpsi / (sdot)
        domega_ds = domega / (sdot)
        d_e_psi_ds = dpsi / (sdot) - kappa(s)
        d_e_y_ds   = (vx * sin(e_psi) + vy * cos(e_psi)) / (sdot)
        d_deltaFz_ds = ddelta_Fz / (sdot)

        f_expl = vertcat(
            d_e_psi_ds,
            d_e_y_ds,
            dx_ds,
            dy_ds,
            dvx_ds,
            dvy_ds,
            dpsi_ds,
            domega_ds,
            d_deltaFz_ds
        )
        f_impl = xdot - f_expl

        # Build AcadosModel
        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = vertcat(s)
        model.name = model_name
        model.x_labels = ["$e_\\psi$", "$e_y$","$x$","$y$","$v_x$","$v_y$","$\\psi$","$r$","$\\Delta F_z$"]
        model.u_labels = ["$a$", "$\\delta$"]
        model.p_labels = ["$s$"]
        return model

    # Race Car Model scaled 1/43, only works on that size of the model, since  needed motor parameters are unknown
    def SpatialDyn_RC(self,kappa):
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
        beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta 
        beta_r = arctan(( params.lr*omega - vy)/(vx+1e-5))
        
        #LATERAL TIRE MODEL
        Fc_f = params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
        Fc_r =  params.Dcr * sin( params.Ccr * arctan(params.Bcr * beta_r) )

        # SECOND ORDER FRICTION TERM
        Fx_d = (params.Cm1 -params.Cm2*vx)*D - params.cr2*vx**2 - params.cr0 * tanh(params.cr3 *vx)

        Fyf = Fc_f*cos(delta)
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
