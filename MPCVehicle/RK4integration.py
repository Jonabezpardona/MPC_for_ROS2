'''
Exlicit Runge Kutta numerical integration of 4th order, not actually used later in the code

Transforms a continuous-time problem into a discrete one by discretizing its domain.
RK4 provides accuracy at a low computational cost, making it a standard
choice among numerical integration methods. Higher order Runge Kutta methods
tend to be more complicated, which makes them lass desirable to work with.

xi+1 = xi +Δt/6 (s1i + 2 s2i + 2 s3i + s4i) 

s1i = f(ti, xi)
s2i = f(ti + Δt/2 , xi + Δt/2 s1i)
s3i = f(ti + Δt/2 , xi + Δt/2 s2i)
s4i = f(ti + Δt, xi + Δts3i)

'''
import numpy as np
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant

# ----------------- RK4 INTEGRATION for KINEMATIC MODEL  ------------------

# ====== Bicycle model in Time Domain ======
class RK4_TimeKinBic:
    def one_step_interation(psi, v, a, beta):
        dx =  v* cos(psi+beta)
        dy =  v * sin(psi+beta)
        dv =  a
        dpsi = v *sin(beta) / lr 
        return dx,dy,dpsi,dv

    def FullIntegrator(xcurrent, u):
        #print("Integration by ERK4")
        h = 0.5
        a = u[0]
        beta = arctan(lr * tan(u[1]) / L) 
        x = xcurrent[0]
        y = xcurrent[1]
        psi = xcurrent[2]
        v  = xcurrent[3]

        dx1,dy1,dpsi1,dv1 = one_step_interation( psi, v, a, beta)
        dx2,dy2,dpsi2,dv2 = one_step_interation( psi + h *dt_sim *dpsi1, v + h *dt_sim *dv1, a, beta)
        dx3,dy3,dpsi3,dv3 = one_step_interation( psi + h *dt_sim *dpsi2, v + h *dt_sim *dv2, a, beta)
        dx4,dy4,dpsi4,dv4 = one_step_interation( psi + dt_sim *dpsi3, v + dt_sim *dv3, a, beta)

        #next
        x_next = x + dt_sim * (dx1+2*dx2+2*dx3+dx4)/6
        y_next = y + dt_sim * (dy1+2*dy2+2*dy3+dy4)/6
        v_next = v + dt_sim * (dv1+2*dv2+2*dv3+dv4)/ 6
        psi_next = psi + dt_sim * (dpsi1+2*dpsi2+2*dpsi3+dpsi4)/ 6
        x_current = np.stack((float(x_next),
        float(y_next),
        float(psi_next),
        float(v_next)))

        return x_current

    def FullIntegratorOld(xcurrent, u):
        print("Integration by hand")
        # RK COEFFICIENTS
        a = 0.5
        b1, b2, b3, b4 = 1/6, 1/3, 1/3, 1/6
        beta = arctan(lr * tan(u[1]) / L)  
        # S2
        x2 = xcurrent[0] + a *dt_sim *xcurrent[2]* cos(xcurrent[3]+beta)
        y2 = xcurrent[1] + a *dt_sim * xcurrent[2] * sin(xcurrent[3]+beta)
        v2 = xcurrent[2] + a *dt_sim * u[0]
        psi2 = xcurrent[3] + a *dt_sim * xcurrent[2] *sin(beta) / lr

        # S3 
        x3 = xcurrent[0] + a *dt_sim * v2* cos(psi2+beta)
        y3 = xcurrent[1] + a *dt_sim * v2* sin(psi2+beta)
        v3 = xcurrent[2] + a *dt_sim * u[0]
        psi3 = xcurrent[3] + a *dt_sim * v2*sin(beta) / lr
        y3 = xcurrent[1] + a *dt_sim * v2* sin(psi2+beta)
        v3 = xcurrent[2] + a *dt_sim * u[0]

        # S4
        x4 = xcurrent[0] + dt_sim * v3* cos(psi3+beta)
        y4 = xcurrent[1] + dt_sim * v3* sin(psi3+beta)
        v4 = xcurrent[2] + dt_sim * u[0]
        psi4 = xcurrent[3] + dt_sim * v3 *sin(beta) / lr

        x = xcurrent[0] + dt_sim * (b1* xcurrent[2]* cos(xcurrent[3]+beta)+b2* v2* cos(psi2+beta)+b3* v3* cos(psi3+beta)+b4* v4* cos(psi4+beta))
        y = xcurrent[1] + dt_sim * (b1* xcurrent[2]* sin(xcurrent[3]+beta)+b2* v2* sin(psi2+beta)+b3* v3* sin(psi3+beta)+b4* v4* sin(psi4+beta))
        v = xcurrent[2] + dt_sim * u[0]
        psi = xcurrent[3] + dt_sim * (b1*xcurrent[2] + b2*v2 + b3*v3 + b4*v4) *sin(beta) / lr

        x_current = np.stack((x,y,v,psi))

        return x_current


# ====== Bicycle model in Spatial Domain ======
class RK4_SpatKinBic:
    def one_step_interation(epsi, ey, psi, v, a, beta):
        sdot = (v *cos(beta) * cos(epsi) - v *sin(beta) *sin(epsi))/(1 - kappa(scurrent)* ey)
        dx =  v* cos(psi+beta)/ sdot
        dy =  v * sin(psi+beta)/ sdot
        dv =  a/ sdot
        dpsi = v *sin(beta) / lr / sdot
        depsi = dpsi - kappa(scurrent) 
        dey = (v *cos(beta) * sin(epsi) + v *sin(beta) * cos(epsi)) / sdot
        return depsi,dey,dx,dy,dpsi,dv 

    def FullIntegrator(xcurrent, u, scurrent):
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
        
        # K1
        depsi1,dey1,dx1,dy1,dpsi1,dv1 = one_step_interation(epsi, ey, psi, v, scurrent, a, beta)
        depsi2,dey2,dx2,dy2,dpsi2,dv2 = one_step_interation(epsi+ h *ds_sim *depsi1, ey+ h *ds_sim *dey1, psi + h *ds_sim *dpsi1, v + h *ds_sim *dv1, scurrent, a, beta)
        depsi3,dey3,dx3,dy3,dpsi3,dv3 = one_step_interation(epsi+ h *ds_sim *depsi2, ey+ h *ds_sim *dey2, psi + h *ds_sim *dpsi2, v + h *ds_sim *dv2, scurrent, a, beta)
        depsi4,dey4,dx4,dy4,dpsi4,dv4 = one_step_interation(epsi+ ds_sim *depsi3, ey+ ds_sim *dey3, psi + ds_sim *dpsi3, v + ds_sim *dv3, scurrent, a, beta)


        #next
        epsi_next = epsi + ds_sim * (depsi1+2*depsi2+2*depsi3+depsi4)/6
        ey_next = ey + ds_sim * (dey1+2*dey2+2*dey3+dey4)/6
        x_next = x + ds_sim * (dx1+2*dx2+2*dx3+dx4)/6
        y_next = y + ds_sim * (dy1+2*dy2+2*dy3+dy4)/6
        v_next = v + ds_sim * (dv1+2*dv2+2*dv3+dv4)/ 6
        psi_next = psi + ds_sim * (dpsi1+2*dpsi2+2*dpsi3+dpsi4)/ 6
        x_current = np.stack((epsi_next,ey_next, x_next,y_next,psi_next, v_next))

        return x_current


# ====== Bicycle Kinematic model in Half Spatial Domain ======
class RK4_HalfSpatKinBic:
    def __init__(self,lr, L, kappa):
        self.lr = lr
        self.L = L
        self.kappa = kappa

    def one_step_interation(self,epsi, ey, psi, v, s, a, beta):
        sdot = (v *cos(beta) * cos(epsi) - v *sin(beta) *sin(epsi))/(1 - self.kappa(s)* ey)
        dx =  v* cos(psi+beta)
        dy =  v * sin(psi+beta)
        dv =  a
        dpsi = v *sin(beta) / self.lr 
        depsi = dpsi - self.kappa(s) * sdot
        dey = (v *cos(beta) * sin(epsi) + v *sin(beta) * cos(epsi)) 
        return depsi,dey,dx,dy,dpsi,dv,sdot

    def FullIntegrator(self,xcurrent, u, dt_sim):
        #print("Integration by ERK4")
        h = 0.5

        a = u[0]
        beta = arctan(self.lr * tan(u[1]) / self.L) 
        
        epsi = xcurrent[0]
        ey = xcurrent[1]
        x = xcurrent[2]
        y = xcurrent[3]
        psi = xcurrent[4]
        v  = xcurrent[5]
        s = xcurrent[6]
        
        depsi1,dey1,dx1,dy1,dpsi1,dv1,ds1 = self.one_step_interation(epsi, ey, psi, v,s, a, beta)
        depsi2,dey2,dx2,dy2,dpsi2,dv2,ds2 = self.one_step_interation(epsi+ h *dt_sim *depsi1, ey+ h *dt_sim *dey1, psi + h *dt_sim *dpsi1, v + h *dt_sim *dv1, s + h *dt_sim *ds1, a, beta)
        depsi3,dey3,dx3,dy3,dpsi3,dv3,ds3 = self.one_step_interation(epsi+ h *dt_sim *depsi2, ey+ h *dt_sim *dey2, psi + h *dt_sim *dpsi2, v + h *dt_sim *dv2, s + h *dt_sim *ds2, a, beta)
        depsi4,dey4,dx4,dy4,dpsi4,dv4,ds4 = self.one_step_interation(epsi+ dt_sim *depsi3, ey+ dt_sim *dey3, psi + dt_sim *dpsi3, v + dt_sim *dv3, s + dt_sim *ds3, a, beta)

        #next
        epsi_next = epsi + dt_sim * (depsi1+2*depsi2+2*depsi3+depsi4)/6
        ey_next = ey + dt_sim * (dey1+2*dey2+2*dey3+dey4)/6
        x_next = x + dt_sim * (dx1+2*dx2+2*dx3+dx4)/6
        y_next = y + dt_sim * (dy1+2*dy2+2*dy3+dy4)/6
        v_next = v + dt_sim * (dv1+2*dv2+2*dv3+dv4)/ 6
        psi_next = psi + dt_sim * (dpsi1+2*dpsi2+2*dpsi3+dpsi4)/ 6
        s_next = s + dt_sim * (ds1+2*ds2+2*ds3+ds4)/ 6
        #epsi_next,ey_next, x_next,y_next,psi_next,v_next, s_nex
        x_current = np.stack((float(epsi_next), float(ey_next), float(x_next), float(y_next), float(psi_next), float(v_next), float(s_next)))

        return x_current
