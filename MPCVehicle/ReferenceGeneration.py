"""
CODE FOR GENERETING ELLIPSE AND SCURVE REFERNCE in TIME and SPATIAL DOMAIN

both TIME AND SPATIAL DOMAIN RETURN trajectory for both kinematic and dynamic model 
it can be chosen which type of trajectoty you want


SPATIAL DOMAIN
idea: based on the coordinates x, y velocity and the sum of heading and slipping angle (actual angle we have),
we compute arc length (S), and we work in spatial domain. thus our controller is NOT dependent on TIME anymore.
based on the values that we have, we can generate curviture value as well.

provides with   x,y position, velocity and the sum of heading and slipping angle (actual angle we have) 
                travel distance needed for executing desired trajectory wrt sampling ds of the controller
                curvature of the trajectory NOTED as kappa

we provide  omega - sampling frequency 
            ds - desired sampling space of controller (travelled distance)
            N_horizon - number of horizon steps s

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline 
from casadi import *

# ------ ELLIPSE GENERATION -----
class EllipseTrajectory:

    def __init__(self):
        self.a = 4.0 
        self.b = 2.2 


        self.omega = 0.3 #0.2

    def Time_Ref(self, omega, dt, N_horizon): 
        t = np.arange(0, 2*np.pi/omega + dt, dt)
        N  = (len(t)-1)
        t_full = np.arange(0, 2*np.pi/omega + N_horizon*dt  + dt, dt) # with horizon for the last point

        x = self.a * np.cos(omega* t_full)
        y = self.b * np.sin(omega * t_full)     
        dx = -self.a *omega* np.sin(omega * t_full)
        dy = self.b *omega* np.cos(omega * t_full)
        v = np.hypot(dx, dy)

        theta = np.arctan2(dy, dx) # psi + beta
        theta = np.unwrap(theta)
        y_ref_kin = np.stack((x, y, v, theta)) # reference for kinematic model


        dtheta = np.gradient(theta, dt) #dtheta    = (dx * ddy - dy * ddx) / (dx**2 + dy**2)

        #we use 0.0 bcs we want our slip angle to be equal to 0; assuming theta = psi 
        vx = v #vx = (dX * np.cos(theta) + dY * np.sin(theta)).flatten() 
        vy = np.zeros(x.shape,) #vy = (dY * np.cos(theta) - dX * np.sin (theta)).flatten() 
        dFz = np.zeros(x.shape) # only if Fz needed

        y_ref_dyn = np.stack((x,y, vx, vy, theta, dtheta, dFz))
        
        return y_ref_kin, y_ref_dyn, N


    def Generating_Sdomain(self): 
        N =  10000
        omega = self.omega
        t = np.linspace(0, 2*np.pi/omega, N)
        dt = t[1] - t[0]

        x = self.a * np.cos(omega* t)
        y = self.b * np.sin(omega * t)     
        dx = -self.a *omega* np.sin(omega * t)
        dy = self.b *omega* np.cos(omega * t)


        ds = np.hypot(dx,  dy) * dt
        s_fine = np.cumsum(ds) # cumulative arc length (left-endpoint integration)

        return s_fine, t

    def Spatial_Ref(self,omega, ds, N_horizon): 
        s_fine, t_fine = self.Generating_Sdomain()
        #print(s_fine)
        s_full = np.arange(0, s_fine[-1] + N_horizon* ds  + ds, ds) # with horizon for the last point
        #print('S full', s_full.shape)
        s0 = np.arange(0, s_fine[-1] + ds, ds)
        S  = (len(s0)-1) * ds
        t = np.interp(s_full, s_fine, t_fine)
        #print(t)

        #omega = self.omega
        X = self.a * np.cos(omega * t) 
        Y = self.b * np.sin(omega * t)     
        dX = -self.a *omega* np.sin(omega * t) 
        dY = self.b *omega* np.cos(omega * t) 
        d2X = -self.a *(omega**2)* np.cos(omega * t) 
        d2Y = -self.b *(omega**2)* np.sin(omega * t) 


        # Spatial derivatives (w.r.t s)
        ds_dt  =  np.hypot(dX, dY)   
        dX_ds = dX / ds_dt 
        dY_ds = dY / ds_dt

        kappa = (dX * d2Y - dY * d2X) / (np.hypot(dX, dY)**3) 

        theta = np.arctan2(dY_ds, dX_ds) 
        theta = np.unwrap(theta)


        #v = np.hypot(vx, vy)
        v = np.ones(s_full.shape) # speed reference depends on the domain, herr in s domain speed represent ds/dt
        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        
        y_ref_kin = np.stack((e_psi, e_y, X, Y, v, theta))


        dtheta= np.gradient(theta, ds) # By definition: dtheta/ds = kappa(s) !! 


        vx =(dX * np.cos(theta) + dY* np.sin(theta)).flatten()
        vy = np.zeros(vx.shape)


        dFz = np.zeros(s_full.shape)

        y_ref_dyn = np.stack((e_psi, e_y, X, Y, vx, vy, theta, dtheta, dFz))


        return y_ref_kin, y_ref_dyn, S, s_full, kappa
 
 

# ------ SCURVE GENERATION -----
class SCurveTrajectory:
    def __init__(self):
        self.x_ctrl = np.array([0, 1, 2, 3, 4, 5,6,7])
        self.y_ctrl = np.array([0, 0.6, 0, -0.6, 0, 0.6 ,0, -0.6])
        self.spline = CubicSpline(self.x_ctrl, self.y_ctrl, bc_type='natural')

    def Time_Ref(self,omega, dt, N_horizon):
        x = np.arange(self.x_ctrl[0], self.x_ctrl[-1], dt)
        y= self.spline(x)
        N  = (len(x)-1 - N_horizon)

        dx = np.gradient(x,dt)
        dy = np.gradient(y,dt)    
        v = np.hypot(dx, dy) 
        theta = np.arctan2(dy, dx)
        
        y_ref_kin = np.stack((x, y, v, theta))
        
        dtheta = np.gradient(theta, dt) #dtheta    = (dx * ddy - dy * ddx) / (dx**2 + dy**2)

        #we use 0.0 bcs we want our slip angle to be equal to 0; assuming theta = psi 
        vx = v #vx = (dX * np.cos(theta) + dY * np.sin(theta)).flatten() 
        vy = np.zeros(x.shape,) #vy = (dY * np.cos(theta) - dX * np.sin (theta)).flatten() 
        dFz = np.zeros(x.shape) # only if Fz needed

        y_ref_dyn = np.stack((x,y, vx, vy, theta, dtheta, dFz))
        

        return y_ref_kin, y_ref_dyn, N

    def Generating_Sdomain(self):
        N = 10000
        x = np.linspace(self.x_ctrl[0], self.x_ctrl[-1], N)
        y= self.spline(x)

        dx = np.gradient(x) 
        dy = np.gradient(y)    
    
        ds= np.hypot(dx, dy)
        s_fine = np.cumsum(ds) #s = np.concatenate([[0], np.cumsum(ds)])
    
        return  s_fine, x


    def Spatial_Ref(self,omega, ds, N_horizon): 
        s_fine, x_fine = self.Generating_Sdomain()
        s0 = np.arange(0, s_fine[-1]- N_horizon*ds+ ds, ds)
        S  = (len(s0)-1)* ds
        s_full = np.arange(0, s_fine[-1] + ds, ds) # with horizon for the last point


        x = np.interp(s_full,s_fine,x_fine)
        y= self.spline(x)
        dx = np.gradient(x,ds) 
        dy = np.gradient(y,ds)
        v = np.ones(s_full.shape) #np.hypot(dx, dy) 
        theta = np.arctan2(dy, dx)
        theta = np.unwrap(theta)

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)

        y_ref_kin = np.stack((e_psi, e_y, x, y,v, theta))

        kappa_num = (dx * np.gradient(dy,ds) - dy * np.gradient(dx,ds))
        kappa = kappa_num / (np.hypot(dx, dy)**3) 

        dtheta= np.gradient(theta, ds) # By definition: dtheta/ds = kappa(s) !! 


        vx =(dx * np.cos(theta) + dy* np.sin(theta)).flatten() # or vx = v
        vy = np.zeros(vx.shape)


        dFz = np.zeros(s_full.shape)

        y_ref_dyn = np.stack((e_psi, e_y, x, y, vx, vy, theta, dtheta, dFz))      
        
        return y_ref_kin, y_ref_dyn, S, s_full, kappa



def plot_trajectory_in_space(y_ref, label='Trajectory', title='Reference Trajectory'):
    plt.figure()
    plt.scatter(y_ref[0,:], y_ref[1,:], label=label)
    plt.xlabel(r'$x [m]$')
    plt.ylabel(r'$y[m]$')
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_trajectory(yref, ds = 0.1, label='Trajectory', title='Reference Trajectory'):
    ny= yref.shape[0]
    N= yref.shape[1]
    timestamps = np.arange(N) * ds
    fig, ax = plt.subplots(ny, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    #labels = ["e_psi", "e_y","x", "y", "heading+slipping angle",  "v", "s_full"] # can be added but is different for spatial/time or kinematic/dynamic
    for i in range(ny):
        ax[i].plot(timestamps, yref[i,:], '--', label='Reference')
        #ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("arc of length [m]")
    plt.tight_layout()
    plt.show()


def plot_kappa(s_full, kappa):

    # Plot kappa shape 
    kappa_interp = interpolant("kappa", "bspline", [s_full], kappa)
    kappa_dense = [float(kappa_interp(s)) for s in s_full]
    
    plt.figure(figsize=(8, 4))
    plt.plot(s_full, kappa, 'o', label='Reference Points')
    plt.plot(s_full, kappa_dense, '-', label='B-spline Interpolation')
    plt.title('B-spline Interpolation of Curvature')
    plt.xlabel('Arc Length s')
    plt.ylabel('Curvature κ(s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    s_curve = SCurveTrajectory()
    ellipse = EllipseTrajectory()
    ds = 0.1
    omega = 0.3
    N_horizon= 10

    y_ref_kin, y_ref_dyn, S, s_full, kappa= s_curve.Spatial_Ref(omega, ds,N_horizon)
    #y_ref_kin, y_ref_dyn, N = s_curve.Time_Ref(omega, ds, N_horizon)
    print(y_ref_kin.shape)
    
    #plot_trajectory_in_space(y_ref_dyn, label='Generated Path') # for time domain
    plot_trajectory_in_space(y_ref_dyn[2:,:], label='Generated Path') # for spatial domain
    #plot_full_trajectory(y_ref_dyn, ds, label='Trajectory', title='Reference Trajectory')
    plot_kappa(s_full, kappa) # only for spatial 

