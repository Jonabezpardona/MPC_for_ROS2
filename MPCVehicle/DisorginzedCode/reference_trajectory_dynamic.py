import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline 
from casadi import *

"""
DYNAMIC BICYCLE MODEL 

our model is based on newton equations, thus instead of being focused on only x,y,psi , we are focused one their derivatives
here since we get X,Y and THETA in global(internal) frame, our derivatives are computed IN GLOBAL FRAME, NOT IN THE CAR FRAME,
thus vx, vy, omega are not the same that we have in the model !!!!


CODE FOR GENERETING ELLIPSE AND SCURVE REFERNCE in TIME and SPATIAL DOMAIN

idea behind spatial: based on the coordinates x, y velocity and the sum of heading and slipping angle (actual angle we have),
we compute arc length (S), and we work in spatial domain. thus our controller is NOT dependent on TIME anymore.
based on the values that we have, we can generate curviture value as well

provides with  x,y position, velocity and the sum of heading and slipping angle (actual angle we have) 
                travel distance needed for executing desired trajectory wrt sampling ds of the controller
                curvature of the trajectory NOTED as kappa

we provide  ds - desired sampling space of controller (travelled distance)
            N_horizon - number of horizon steps s
"""

# REFERENCES FOR DYNAMIC MODEL

# ------ STRAIGHT LINE -----

class StraightLineTrajectory:
    def __init__(self):
        self.Str_line = 20.0

    def time_ref(self, omega, dt, N_horizon):
        t0 = np.arange(0, self.Str_line+ omega, omega)
        T  = (len(t0)-1)*dt

        t_full = np.arange(0, self.Str_line+omega*N_horizon + omega, omega) # with horizon for the last point
        kappa = np.zeros(t_full.shape)
        # going on x axis
        x = t_full
        vx = np.gradient(x,dt)
        y = np.zeros(x.shape)
        vy = np.zeros(x.shape)
        psi = np.zeros(x.shape)
        dpsi = np.zeros(x.shape)
        y_ref = np.stack((x, y, vx,vy, psi, dpsi))
        y_ref0 = np.hstack((x[0], y[0], vx[0], vy[0], psi[0],dpsi[0]))

        return y_ref,y_ref0, T

    def spatial_ref(self, ds, N_horizon):
        s0 = np.arange(0, self.Str_line+ ds, ds)
        S  = (len(s0)-1)*ds

        s_full = np.arange(0, self.Str_line+ds*N_horizon + ds, ds) # with horizon for the last point
        kappa = np.zeros(s_full.shape)
        # going on x axis
        x = s_full
        y = np.zeros(x.shape)
        vx = np.gradient(x,ds)# ds * np.ones(x.shape)
        vy =np.zeros(x.shape)
        psi = np.zeros(x.shape)
        dpsi = psi
        e_psi = psi
        e_y = psi
        
        y_ref = np.stack((e_psi, e_y, x, y, vx, vy, psi, dpsi))
        

        return y_ref, S, s_full, kappa

# ------ ELLIPSE GENERATION -----
class EllipseTrajectory:
    def __init__(self):
        self.a  = 4.0
        self.b =2.2
        self.a  = 10.0
        self.b =8.0
        #self.a  = 1.2
        #self.b  = 0.7
        self.omega = 0.2
        
        

    def time_ref(self, omega, dt, N_horizon): 
        t = np.arange(0, 2*np.pi/omega + dt, dt)
        T  = (len(t)-1)*dt
        t_full = np.arange(0, 2*np.pi/omega + N_horizon*dt  + dt, dt) # with horizon for the last point

        x = self.a * np.cos(omega* t_full)
        y = self.b * np.sin(omega * t_full)     
        dX = -self.a *omega* np.sin(omega * t_full) # x speed in global coordinate system
        dY = self.b *omega* np.cos(omega * t_full) # y speed in global coordinate system
        #d2X = -self.a *(omega**2)* np.cos(omega * t) 
        #d2Y = -self.b *(omega**2)* np.sin(omega * t) 
        v = np.hypot(dX, dY)

        theta = np.arctan2(dY, dX) # psi + beta
        theta = np.unwrap(theta) # derivative psi + beta
        dtheta = np.gradient(theta, dt) #
        #dtheta    = (dx * ddy - dy * ddx) / (dx**2 + dy**2)
        #dtheta  = (dx * ddy - dy * ddx) / (np.hypot(dx, dy)**3) # should try it with this as well
     
        
        # assuming theta = psi 
        vx = (dX * np.cos(theta) + dY * np.sin(theta)).flatten() #dX * cos(theta) + dY * sin (theta)
        vy = (dY * np.cos(theta) - dX * np.sin (theta)).flatten() # instead of this we use 0.0 bcs we want our slip angle to be equal to 0
        vy0 = np.zeros(x.shape,)
        dFz = np.zeros(x.shape)
        y_ref = np.stack((x,y,vx, vy, theta, dtheta, dFz))
        y_ref0 = np.hstack((x[0], y[0], vx[0], vy[0], theta[0],dtheta[0], dFz[0]))
        return y_ref,y_ref0, T


    def time_ref_worldframe(self, omega, dt, N_horizon): 
        t = np.arange(0, 2*np.pi/omega + dt, dt)
        T  = (len(t)-1)*dt
        t_full = np.arange(0, 2*np.pi/omega + N_horizon*dt  + dt, dt) # with horizon for the last point

        x = self.a * np.cos(omega* t_full)
        y = self.b * np.sin(omega * t_full)     
        dX = -self.a *omega* np.sin(omega * t_full) # x speed in internal coordinate system
        dY = self.b *omega* np.cos(omega * t_full) # y speed in internal coordinate system
        v = np.hypot(dX, dY)

        theta = np.arctan2(dY, dX) # psi + beta
        theta = np.unwrap(theta) # derivative psi + beta

        dtheta= np.gradient(theta, t_full) #omega   = (dx * ddy - dy * ddx) / (dx**2 + dy**2)
        y_ref = np.stack((x,y,dX, dY, theta, dtheta))
        y_ref0 = np.hstack((x[0], y[0], v[0], 0.0, theta[0],dtheta[0]))
        return y_ref,y_ref0, T

    def generating_spatial_domain(self): 
        N =  100000
        omega = self.omega
        t = np.linspace(0, 2*np.pi/omega, N)
        dt = t[1] - t[0]
        print(dt)

        x = self.a * np.cos(omega* t)
        y = self.b * np.sin(omega * t)     
        dx = -self.a *omega* np.sin(omega * t)
        dy = self.b *omega* np.cos(omega * t)

        # Compute cumulative arc length (left-endpoint integration)
        ds = np.hypot(dx,  dy) * dt
        #s_fine = np.zeros(N)
        #for i in range(1, N):
        #    s_fine[i] = s_fine[i-1] + ds[i-1] * dt
        s_fine = np.cumsum(ds)
        S_total = s_fine[-1]
        return S_total,s_fine, t

    def spatial_ref(self,ds, N_horizon): 
        S_total, s_fine, t_fine = self.generating_spatial_domain()

        s_full = np.arange(0, S_total + N_horizon*ds  + ds, ds) # with horizon for the last point
        s0 = np.arange(0, S_total+ ds, ds)
        S  = (len(s0)-1)*ds
        
        
        t = np.interp(s_full, s_fine, t_fine)
        dt = t[1]-t[0]
        dt_ds = np.gradient(t, s_full)


        omega = self.omega
        X = self.a * np.cos(omega * t) 
        Y = self.b * np.sin(omega * t)     
        dX = -self.a *omega* np.sin(omega * t) 
        dY = self.b *omega* np.cos(omega * t) 
        d2X = -self.a *(omega**2)* np.cos(omega * t) 
        d2Y = -self.b *(omega**2)* np.sin(omega * t) 

        #v = ds/dt#np.hypot(dX, dY)  # ds/dt
        print(dt)

        # Spatial derivatives (w.r.t s)
        ds_dt  =   np.hypot(dX, dY)   
        dX_ds = dX / ds_dt 
        dY_ds = dY / ds_dt

        theta = np.arctan2(dY_ds, dX_ds) 
        theta = np.unwrap(theta)
        dtheta= np.gradient(theta, ds) # By definition: dtheta/ds = kappa(s) !! 
        kappa = (dX * d2Y - dY * d2X) / (np.hypot(dX, dY)**3) 
        vx =(dX * np.cos(theta) + dY* np.sin(theta)).flatten()# ds_dt # 0.8*np.ones(s_full.shape) #ds_dt# # #dX * cos(theta) + dY * sin (theta) #ds_dt#(
        vy = np.zeros_like(vx)#((dY_ds * np.cos(theta) - dX_ds * np.sin (theta))).flatten() # instead of this we use 0.0 bcs we want our slip angle to be equal to 0

        #v = np.hypot(vx, vy)
        #vx =  vx/v #dX * cos(theta) + dY * sin (theta)
        #vy = vy/v
 

        
        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.stack((e_psi, e_y, X, Y, vx, vy, theta, dtheta))
        y_ref = np.vstack((e_psi, e_y, X, Y, vx, vy, theta,dtheta)).T


        return y_ref, S, s_full, kappa
 
    def halfspatial_offline_ref(self,ds, N_horizon): 
        S_total, s_fine, kappa_fine, t_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total + N_horizon*ds  + ds, ds) # with horizon for the last point

        t = np.interp(s_full, s_fine, t_fine)
        dt = t[1]-t[0]
        #kappa = np.interp(s_full, s_fine, kappa_fine) #*dt/ds

        omega = self.omega
        x = self.a * np.cos(omega * t) 
        y = self.b * np.sin(omega * t)     
        dx = -self.a *omega* np.sin(omega * t) 
        dy = self.b *omega* np.cos(omega * t) 
        d2x = -self.a *(omega**2)* np.cos(omega * t) 
        d2y = -self.b *(omega**2)* np.sin(omega * t) 
        theta = np.arctan2(dy, dx) 
        theta = np.unwrap(theta)
        dtheta= np.gradient(theta, dt)

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.stack((e_psi, e_y, x, y, dx, dy, theta, dtheta, s_full))

        kappa = (dx * d2y - dy * d2x) / (np.hypot(dx, dy)**3) 

        return y_ref, S, s_full, kappa

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
        self.x_ctrl = np.array([0, 10, 20, 30, 40, 50, 60, 70])
        self.y_ctrl = np.array([0, 3, 0, -3, 0, 3, 0, -3 ])
        #self.x_ctrl = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])# , 2.1, 2.4]) # FOR RACE CARS 
        #self.y_ctrl = np.array([0, 0.07, 0, -0.07, 0, 0.07, 0]) #, -0.1, 0]) # FOR RACE CARS
        self.spline = CubicSpline(self.x_ctrl, self.y_ctrl, bc_type='natural')

    def time_ref(self,omega,dt, N_horizon):
        x = np.arange(self.x_ctrl[0], self.x_ctrl[-1], omega)
        y= self.spline(x)
        dX = np.gradient(x,dt)
        dY = np.gradient(y,dt)  
          

        theta = np.arctan2(dY, dX)
        theta = np.unwrap(theta)
        dtheta= np.gradient(theta,dt)

        vx = (dX * np.cos(theta) + dY * np.sin(theta)).flatten() #dX * cos(theta) + dY * sin (theta)
        vy = (dY * np.cos(theta) - dX * np.sin (theta)).flatten() # instead of this we use 0.0 bcs we want our slip angle to be equal to 0
        
        dFz = np.zeros(x.shape)
        y_ref = np.stack((x, y,vx,vy,theta, dtheta))
        y_ref0 = np.hstack((x[0], y[0], vx[0], vy[0], theta[0],dtheta[0]))

        T  = (len(x)-1 - N_horizon)*dt
        return y_ref, y_ref0 , T


    def generating_spatial_domain(self):
        N = 10000
        x = np.linspace(self.x_ctrl[0], self.x_ctrl[-1], N)
        y= self.spline(x)

        dx = np.gradient(x) 
        dy = np.gradient(y)    

        #ds= np.hypot(np.diff(x), np.diff(y))
        ds= np.hypot(dx, dy)
        #s = np.concatenate([[0], np.cumsum(ds)])
        s = np.cumsum(ds)
        S_total = s[-1]
        kappa_num = (dx * np.gradient(dy) - dy * np.gradient(dx))
        kappa = kappa_num / (ds**3)  # curvature of the trajectory

        return S_total, s, kappa, x

    def spatial_S_ref(self,ds, N_horizon):
        S_total, s_fine, kappa_fine , _ = self.generating_spatial_domain()
        s0 = np.arange(0, S_total- N_horizon*ds + ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total + ds, ds) # with horizon for the last point

        kappa = np.interp(s_full, s_fine, kappa_fine)

        return S, s_full, kappa

    def spatial_ref(self, ds, N_horizon): 

        S_total, s_fine, kappa_fine,x_fine = self.generating_spatial_domain()
        s0 = np.arange(0, S_total- N_horizon*ds+ ds, ds)
        S  = (len(s0)-1)*ds
        s_full = np.arange(0, S_total   + ds, ds) # with horizon for the last point
        #kappa = np.interp(s_full, s_fine, kappa_fine) 

        x = np.interp(s_full,s_fine,x_fine)
        y= self.spline(x)
        dX = np.gradient(x,ds) 
        print(dX)
        dY = np.gradient(y,ds)
        print(dY)
        theta = np.arctan2(dY, dX)
        theta = np.unwrap(theta)
        dtheta= np.gradient(theta, ds)

        ds_dt  =   np.hypot(dX, dY)
        vx =2*(dX * np.cos(theta) + dY * np.sin(theta)).flatten()# #2*np.ones(s_full.shape) #0.6*np.ones(s_full.shape)  #0.8*(dX * np.cos(theta) + dY * np.sin(theta)).flatten() #dX * cos(theta) + dY * sin (theta)
        vy = (dY * np.cos(theta) - dX* np.sin (theta)).flatten() # instead of this we use 0.0 bcs we want our slip angle to be equal to 0

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.vstack((e_psi, e_y, x, y, vx, vy, theta,dtheta)).T
        #y_ref = np.stack((e_psi, e_y, x, y, vx, vy, theta,dtheta))

        kappa_num = (dX * np.gradient(dY,ds) - dY * np.gradient(dX,ds))
        kappa = kappa_num / (np.hypot(dX, dY)**3) 
        
        return y_ref, S, s_full, kappa

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
        theta = np.arctan2(dy, dx)
        theta = np.unwrap(theta)
        dtheta= np.gradient(theta, dt)  

        e_psi = np.zeros(s_full.shape)
        e_y = np.zeros(s_full.shape)
        y_ref = np.stack((e_psi, e_y, x, y, dx, dy, theta, dtheta, s_full))

        kappa_num = (dx * np.gradient(dy,ds) - dy * np.gradient(dx,ds))
        kappa = kappa_num / (np.hypot(dx, dy)**3) 
        
        return y_ref, S, s_full, kappa


def plot_trajectory_in_space(y_ref, label='Trajectory', title='Reference Trajectory'):
    plt.figure()
    # time
    #plt.scatter(y_ref[0,:], y_ref[1,:], label=label)
    #spatial
    plt.scatter(y_ref[2,:], y_ref[3,:], label=label)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_trajectory_time(yref, ds = 0.1, label='Trajectory', title='Reference Trajectory'):
    ny= yref.shape[0]
    N= yref.shape[1]
    timestamps = np.arange(N) * ds
    fig, ax = plt.subplots(ny, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y","x", "y", "vx", "vy", "heading+slipping angle", "omega"]
    for i in range(ny):
        ax[i].plot(timestamps, yref[i,:], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

def plot_full_trajectory_spatial(yref, ds = 0.1, label='Trajectory', title='Reference Trajectory'):
    ny= yref.shape[0]
    N= yref.shape[1]
    timestamps = np.arange(N) * ds
    fig, ax = plt.subplots(ny, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y","x", "y", "vx", "vy", "heading+slipping", "omega"]#, "s_full"]
    for i in range(ny):
        ax[i].plot(timestamps, yref[i,:], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("arc of length [m]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ellipse = EllipseTrajectory()
    scurve = SCurveTrajectory()
    straight = StraightLineTrajectory()
    y_ref, S, s_full, kappa= ellipse.spatial_ref(0.1,20)
    print(f"y_ref: {y_ref.shape}")
    plot_trajectory_in_space(y_ref, label='Ellipse Path')
    plot_full_trajectory_spatial(y_ref, ds = 0.1, label='Trajectory', title='Reference Trajectory')


    '''
    s_curve = SCurveTrajectory()
    #S,s0,s_full, kappa, y_ref = s_curve.spatial_offline_ref(0.1,20)
    ellipse = EllipseTrajectory()
    S, s0, s_full, kappa, y_ref = ellipse.spatial_offline_ref(0.1,20)

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
    '''
