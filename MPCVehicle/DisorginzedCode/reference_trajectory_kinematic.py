import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline 
from casadi import *
"""
CODE FOR GENERETING ELLIPSE AND SCURVE REFERNCE in SPATIAL DOMAIN

idea: based on the coordinates x, y velocity and the sum of heading and slipping angle (actual angle we have),
we compute arc length (S), and we work in spatial domain. thus our controller is NOT dependent on TIME anymore.
based on the values that we have, we can generate curviture value as well

provides with   x,y position, velocity and the sum of heading and slipping angle (actual angle we have) 
                travel distance needed for executing desired trajectory wrt sampling ds of the controller
                curvature of the trajectory NOTED as kappa

we provide  ds - desired sampling space of controller (travelled distance)
            N_horizon - number of horizon steps s
"""

# REFERENCES FOR KINEMATIC MODEL

# ------ STRAIGHT LINE -----

class StraightLineTrajectory:
    def __init__(self):
        self.S_line = 50.0

    def spatial_ref(self, ds, N_horizon):
        s0 = np.arange(0, self.S_line+ ds, ds)
        S  = (len(s0)-1)*ds

        s_full = np.arange(0, self.S_line+ds*N_horizon + ds, ds) # with horizon for the last point
        kappa = np.zeros(s_full.shape)
        # going on x axis
        x = s_full
        y = np.zeros(x.shape)
        psi = np.zeros(x.shape)
        e_psi = psi
        e_y = psi
        y_ref = np.vstack((e_psi, e_y, x, y, psi)).T

        return S, s0, s_full, kappa, y_ref

# ------ ELLIPSE GENERATION -----
class EllipseTrajectory:
    def __init__(self):
        self.a = 4.0
        self.b = 2.2
        self.omega = 0.3

    def time_ref(self, omega, dt, N_horizon): 
        t = np.arange(0, 2*np.pi/omega + dt, dt)
        T  = (len(t)-1)*dt
        t_full = np.arange(0, 2*np.pi/omega + N_horizon*dt  + dt, dt) # with horizon for the last point

        x = self.a * np.cos(omega* t_full)
        y = self.b * np.sin(omega * t_full)     
        dx = -self.a *omega* np.sin(omega * t_full)
        dy = self.b *omega* np.cos(omega * t_full)
        v = np.hypot(dx, dy)

        theta = np.arctan2(dy, dx) # psi + beta
        theta = np.unwrap(theta)

        y_ref = np.stack((x, y, v, theta))
        return y_ref, T

    def generating_spatial_domain(self): 
        N =  10000
        omega = self.omega
        t = np.linspace(0, 2*np.pi/omega, N)
        dt = t[1] - t[0]
        print(dt)
        

        x = self.a * np.cos(omega* t)
        y = self.b * np.sin(omega * t)     
        dx = -self.a *omega* np.sin(omega * t)
        dy = self.b *omega* np.cos(omega * t)
        v = np.hypot(dx, dy)

        theta = np.arctan2(dy, dx) # psi + beta
        theta = np.unwrap(theta)

        # compute derivatives and arc length differentials
        ds = np.hypot(dx,  dy)
        d2x = -self.a *(omega**2)* np.cos(omega * t)
        d2y = -self.b *(omega**2)* np.sin(omega * t)
        kappa = (dx * d2y - dy * d2x) / (ds**3)

        # Compute cumulative arc length (left-endpoint integration)
        s_fine = np.zeros(N)
        for i in range(1, N):
            s_fine[i] = s_fine[i-1] + ds[i-1] * dt
        S_total = s_fine[-1]
        return S_total,s_fine, kappa, t

    def spatial_ref(self,ds, N_horizon): 
        S_total, s_fine, kappa_fine, t_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total + N_horizon*ds  + ds, ds) # with horizon for the last point

        t = np.interp(s_full, s_fine, t_fine)
        dt = t[1]-t[0]
        print(dt/ds) 
        #kappa = np.interp(s_full, s_fine, kappa_fine) #*dt/ds

        omega = self.omega
        x = self.a * np.cos(omega * t) 
        y = self.b * np.sin(omega * t)     
        dx = -self.a *omega* np.sin(omega * t) 
        dy = self.b *omega* np.cos(omega * t) 
        d2x = -self.a *(omega**2)* np.cos(omega * t) 
        d2y = -self.b *(omega**2)* np.sin(omega * t) 
        v = np.ones(s_full.shape) #np.hypot(dx, dy) # # speed reference depends on the domain
        theta = np.arctan2(dy, dx) 
        theta = np.unwrap(theta)

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.vstack((e_psi, e_y, x, y, theta, v)).T

        kappa = (dx * d2y - dy * d2x) / (np.hypot(dx, dy)**3) 

        return S, s0, s_full, kappa, y_ref
 
    def halfspatial_offline_ref(self,ds, N_horizon): 
        S_total, s_fine, kappa_fine, t_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total + N_horizon*ds  + ds, ds) # with horizon for the last point

        t = np.interp(s_full, s_fine, t_fine)
        dt = t[1]-t[0]
        print(dt/ds) 
        #kappa = np.interp(s_full, s_fine, kappa_fine) #*dt/ds

        omega = self.omega
        x = self.a * np.cos(omega * t) 
        y = self.b * np.sin(omega * t)     
        dx = -self.a *omega* np.sin(omega * t) *dt/ds
        dy = self.b *omega* np.cos(omega * t) *dt/ds
        d2x = -self.a *(omega**2)* np.cos(omega * t) *(dt/ds)**2
        d2y = -self.b *(omega**2)* np.sin(omega * t) *(dt/ds)**2
        v =  dt/ds#np.hypot(dx, dy) #3*np.ones(s_full.shape)# speed reference depends on the domain
        theta = np.arctan2(dy, dx) 
        theta = np.unwrap(theta)

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.vstack((e_psi, e_y, x, y, theta, v, s_full)).T

        kappa = (dx * d2y - dy * d2x) / (np.hypot(dx, dy)**3) 

        return S, s0, s_full, kappa, y_ref

    def spatial_s_ref(self, ds, N_horizon):
        S_total, s_fine, kappa_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total + N_horizon*ds  + ds, ds) # with horizon for the last point

        kappa = np.interp(s_full, s_fine, kappa_fine)

        return S, s_full, kappa

# ------ SCURVE GENERATION -----
class SCurveTrajectory:
    def __init__(self):
        self.x_ctrl = np.array([0, 1, 2, 3, 4, 5,6,7])
        self.y_ctrl = np.array([0, 0.6, 0, -0.6, 0, 0.6 ,0, -0.6])
        self.spline = CubicSpline(self.x_ctrl, self.y_ctrl, bc_type='natural')

    def time_ref(self,dt, N_horizon):
        x = np.arange(self.x_ctrl[0], self.x_ctrl[-1], dt)
        y= self.spline(x)
        dx = np.gradient(x,dt)
        dy = np.gradient(y,dt)    
        v = np.hypot(dx, dy) 
        theta = np.arctan2(dy, dx)
        
        y_ref = np.stack((x, y, v, theta))
        T  = (len(x)-1 - N_horizon)*dt
        return y_ref, T


    def generating_spatial_domain(self):
        N = 10000
        x = np.linspace(self.x_ctrl[0], self.x_ctrl[-1], N)
        y= self.spline(x)

        dx = np.gradient(x) 
        print(dx)
        dy = np.gradient(y)    
        print(dy)
        v = np.hypot(dx, dy) 
        theta = np.arctan2(dy, dx)
        dtheta= np.gradient(theta)

        #ds= np.hypot(np.diff(x), np.diff(y))
        ds= np.hypot(dx, dy)
        #s = np.concatenate([[0], np.cumsum(ds)])
        s = np.cumsum(ds)
        print(s.shape)
        S_total = s[-1]
        kappa_num = (dx * np.gradient(dy) - dy * np.gradient(dx))
        kappa = kappa_num / (ds**3)  # curvature of the trajectory
        print(kappa.shape)
        return S_total, s, kappa, x

    def spatial_ref(self,ds, N_horizon):
        S_total, s_fine, kappa_fine , _ = self.generating_spatial_domain()
        s0 = np.arange(0, S_total- N_horizon*ds + ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total + ds, ds) # with horizon for the last point

        kappa = np.interp(s_full, s_fine, kappa_fine)

        return S, s_full, kappa

    def spatial_offline_ref(self,ds, N_horizon): 
        S_total, s_fine, kappa_fine,x_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total- N_horizon*ds+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total   + ds, ds) # with horizon for the last point
        #kappa = np.interp(s_full, s_fine, kappa_fine) 

        x = np.interp(s_full,s_fine,x_fine)
        y= self.spline(x)
        dx = np.gradient(x,ds) 
        dy = np.gradient(y,ds)
        v = np.ones(s_full.shape) #np.hypot(dx, dy) 
        theta = np.arctan2(dy, dx)
        theta = np.unwrap(theta)

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.vstack((e_psi, e_y, x, y, theta,v)).T

        kappa_num = (dx * np.gradient(dy,ds) - dy * np.gradient(dx,ds))
        kappa = kappa_num / (np.hypot(dx, dy)**3) 
        
        return S,s0,s_full, kappa, y_ref

    def halfspatial_offline_ref(self,ds, N_horizon): 
        S_total, s_fine, kappa_fine,x_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total- N_horizon*ds+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total   + ds, ds) # with horizon for the last point
        #kappa = np.interp(s_full, s_fine, kappa_fine) 

        x = np.interp(s_full,s_fine,x_fine)
        y= self.spline(x)
        dx = np.gradient(x,ds) 
        dy = np.gradient(y,ds)
        v = np.ones(s_full.shape)  #np.hypot(dx, dy) 
        theta = np.arctan2(dy, dx)
        theta = np.unwrap(theta)

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.vstack((e_psi, e_y, x, y, theta,v,s_full)).T

        kappa_num = (dx * np.gradient(dy,ds) - dy * np.gradient(dx,ds))
        kappa = kappa_num / (np.hypot(dx, dy)**3) 
        
        return S,s0,s_full, kappa, y_ref


def plot_trajectory_in_space(y_ref, label='Trajectory', title='Reference Trajectory'):
    plt.figure()
    plt.scatter(y_ref[:,2], y_ref[:,3], label=label)
    plt.xlabel(r'$x [m]$')
    plt.ylabel(r'$y[m]$')
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_trajectory(yref, ds = 0.1, label='Trajectory', title='Reference Trajectory'):
    ny= yref.shape[1]
    N= yref.shape[0]
    timestamps = np.arange(N) * ds
    fig, ax = plt.subplots(ny, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y","x", "y", "heading+slipping angle",  "v", "s_full"]
    for i in range(ny):
        ax[i].plot(timestamps, yref[:,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("arc of length [m]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    s_curve = SCurveTrajectory()
    #S,s0,s_full, kappa, y_ref = s_curve.spatial_offline_ref(0.1,20)
    ellipse = EllipseTrajectory()
    S, s0, s_full, kappa, y_ref = ellipse.spatial_ref(0.1,20)

    plot_trajectory_in_space(y_ref, label='Ellipse Path')
    plot_full_trajectory(y_ref, ds = 0.1, label='Trajectory', title='Reference Trajectory')

    kappa_interp = interpolant("kappa", "bspline", [s_full], kappa)
    kappa_dense = [float(kappa_interp(s)) for s in s_full]
    # Plot
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

