'''
ANIMATION OF THE CAR ON THE REFERENCE TRAJECTORY ALONG WITH ITS FEATURES

CLASS  SimulationAnimator updates the car at each sampling time of the simulatior, 
it can updated either in real time (cannot be saved) or saved and proposed after the trajectory is done

other function represent a reference trajectory in space and by each state
'''
import numpy as np
import time
from math import cos, sin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_ref_trajectory(y_ref):
    timestampsy = np.linspace(0,(N+1)*ds_ocp,N+1)
    plt.figure()
    plt.plot(y_ref[:N+1,2], y_ref[:N+1,3], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()
    plt.show()

def plot_states(simX, simU, y_ref,ds_sim,ds_ocp):
    Nsim = simX.shape[0] - 1
    N = Nsim
    timestampsx = np.linspace(0,(Nsim+1)*ds_sim,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*ds_sim,Nsim)
    #timestampsy = np.linspace(0,(N+1)*ds_ocp,N+1)

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Distance', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y", "x", "y", "theta", "v", "a", "delta"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:, i])
        #ax[i].plot(timestampsy, y_ref[:N+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx-1].plot(timestampsu, simU[:, i])
        ax[i + nx-1].set_ylabel(labels[i + nx-1])
    ax[-1].set_xlabel("travelled distance [m]")
    plt.tight_layout()
    plt.show()


class SimulationAnimator:
    # --- plots the refenence and the road and sets up the fiugre  ---
    def __init__(self, x_ref, y_ref,psi, lf, lr,w = 1.2, wheel_length=0.4, wheel_width=0.15):
        self.x_data = []
        self.y_data = []

        # === Vehicle dimensions === 
        self.lf = lf
        self.lr = lr
        self.car_width = w
        self.car_length = lr + lf
        self.wheel_length= wheel_length
        self.wheel_width = wheel_width

        # ====== Road setup ======
        self.half_road_width = 1.0
        dx = self.half_road_width  * np.sin(psi)
        dy = self.half_road_width * -np.cos(psi)

        left_x = x_ref + dx
        left_y = y_ref+ dy
        right_x = x_ref - dx
        right_y = y_ref - dy

        road_x = np.concatenate([left_x, right_x[::-1]])
        road_y = np.concatenate([left_y, right_y[::-1]])

        # Setup figure
        self.fig, self.ax = plt.subplots()
        self.ax.plot(x_ref, y_ref, '--', alpha=0.9 , c = "gold")#,label='Reference Trajectory'
        self.ax.fill(road_x, road_y, color="gray", alpha=0.75)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x[m]', fontsize=14)
        self.ax.set_ylabel('y[m]', fontsize=14)
        self.ax.set_title("Trajectory with Vehicle")
        self.ax.legend()
        self.time_text = self.ax.text(0.95, 0.95, '', transform=self.ax.transAxes,
                                      ha='right', va='top', fontsize=12)
        self.speed_text = self.ax.text(0.05, 0.02, '', transform=self.ax.transAxes,
                                       ha='left', va='bottom', fontsize=12)

        # Objects to be updated
        self.body_path, = self.ax.plot([], [], color="tab:gray", linewidth=2, zorder=0)
        self.body_dot = self.ax.scatter([], [], color="tab:gray", s=50, zorder=2)
        self.body_line, = self.ax.plot([], [], color="tab:gray", linewidth=3, zorder=1)
        self.wheel_f = None
        self.wheel_r = None
        self.safety_circle = None
        self.car_body = None
        self.pos_text = None
        self.wheels = []

        
        plt.ion()
        plt.show()

    # --- updates car and wheel position  ---
    def update_car(self, x, y, psi, delta):
        # Remove old patches
        for wheel in self.wheels:
            wheel.remove()
        self.wheels.clear()
        if self.car_body:
            self.car_body.remove()
        if self.safety_circle:
            self.safety_circle.remove()


        # === Car body ===
        car_center_x = x
        car_center_y = y
        car_bottom_left_x = car_center_x - self.car_length / 2
        car_bottom_left_y = car_center_y - self.car_width / 2

        # Rectangle with rotation
        self.car_body = patches.Rectangle(
            (-self.car_length/2, -self.car_width/2),
            self.car_length,
            self.car_width,
            edgecolor='black',
            facecolor='darkgray',
            lw=2
        )
        trans = transforms.Affine2D().rotate_around(0, 0, psi).translate(x, y) + self.ax.transData
        self.car_body.set_transform(trans)
        self.ax.add_patch(self.car_body)

        # === Wheel positions ===
        wheel_coords = [
            (+0.3 * self.car_length, +0.5 * self.car_width, delta),  # Front Left
            (+0.3 * self.car_length, -0.5 * self.car_width, delta),  # Front Right
            (-0.3 * self.car_length, +0.5 * self.car_width, 0),      # Rear Left
            (-0.3 * self.car_length, -0.5 * self.car_width, 0),      # Rear Right
        ]

        for dx, dy, angle in wheel_coords:
            # Rotate around center
            wx = x + cos(psi)*dx - sin(psi)*dy
            wy = y + sin(psi)*dx + cos(psi)*dy
            wheel_angle = np.degrees(psi + angle)

            wheel = patches.Rectangle(
                (-self.wheel_length/2, -self.wheel_width/2),
                self.wheel_length,
                self.wheel_width,
                color='gray'
            )
            wheel_trans = transforms.Affine2D().rotate_deg_around(0, 0, wheel_angle).translate(wx, wy) + self.ax.transData
            wheel.set_transform(wheel_trans)
            self.ax.add_patch(wheel)
            self.wheels.append(wheel)

        # === Safety circle ===
        self.safety_circle = patches.Circle((x, y), 1.2, color="tab:red", alpha=0.2)
        self.ax.add_patch(self.safety_circle)

    # --- plots/draws a car on the same plot in RT thus cannot be saved or reproduced later ---
    def plot_car_RT(self, x, y, psi,v, delta,t):
        self.x_data.append(x)
        self.y_data.append(y)

        self.update_car(x, y, psi, delta)

        # === Update speed and time text ===
        self.time_text.set_text(f"{t:.2f}s")
        self.speed_text.set_text(f"Speed: {v:.2f} m/s")

        # === Update car position text ===
        if self.pos_text:
            self.pos_text.remove()
        self.pos_text = self.ax.text(x, y + 1.5, f"{x:.1f}, {y:.1f}", fontsize=9)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_trajectory(self, simX,Nf,dt_sim):
        timestampsx = np.linspace(0,(Nf+1)*dt_sim,Nf+1)
        self.ax.plot(simX[:Nf+1,2],simX[:Nf+1,3], label='Simulation')
        plt.ion()
        plt.show()
        
  
    def update_anim(self, frame, simX, simY,simP,simV,simD, times):
        # unpack state & control
        x, y, psi, v = simX[frame], simY[frame], simP[frame], simV[frame]
        delta        = simD[frame]
        t            = times[frame]

        self.update_car(x, y, psi, delta)

        # Texts
        self.time_text.set_text(f"{t:.2f}s")
        self.speed_text.set_text(f"Speed: {v:.2f} m/s")
        if self.pos_text is None:
            self.pos_text = self.ax.text(x, y + 1.5, f"{x:.1f}, {y:.1f}", fontsize=9)
        else:
            self.pos_text.set_position((x, y + 1.5))
            self.pos_text.set_text(f"{x:.1f}, {y:.1f}")


        # keep camera centered
        #self.ax.set_xlim(x - 5, x + 5)
        #self.ax.set_ylim(y - 3, y + 3)

        return []
    
    # --- 3) Create an update function for FuncAnimation ---
    def animate(self,simX, simY,simP,simV,simD,t):
        N = len(t)
        dt = (t[1]-t[0])
        # --- 4) Build the FuncAnimation ---
        anim = FuncAnimation(self.fig,
                            lambda f: self.update_anim(f, simX, simY,simP,simV,simD, t),
                            frames=N, blit=False, interval=dt*1000)

 
        # GIF via Pillow
        gif_writer = PillowWriter(fps=(1/dt))
        anim.save("simulation_RT_ellipse.gif", writer=gif_writer)
        print("Animation saved!")
        return anim

    def close(self):
        plt.ioff()
        plt.close(self.fig)

if __name__ == "__main__":
    t_ref = np.linspace(0, 6.3, 63)
    x_ref = 7*np.cos(t_ref)
    y_ref = 7*np.sin(t_ref)
    psi = np. arctan2(np.gradient(y_ref), np.gradient(x_ref))

    lf = 1.5
    lr = 1.0

    # test for plot_car_RT
    anim = SimulationAnimator(x_ref, y_ref, psi, lf, lr)
    #anim.plot_car_RT(x_ref[0], y_ref[0], np.pi/2, 0, 0.1, 0)

    simX = np.zeros(t_ref.shape)
    simY = np.zeros(t_ref.shape)
    simV = np.zeros(t_ref.shape)
    simP = np.zeros(t_ref.shape)
    simD = np.zeros(t_ref.shape)


    delta = 0.2  # upravljački ugao točkova
    for i in range(len(t_ref)):
        # U realnosti bi ovde koristio MPC i simulaciju za px, py, psi, delta
        px = x_ref[i]
        py = y_ref[i]
        psi = np.arctan2(np.gradient(y_ref)[i], np.gradient(x_ref)[i])  # približna orijentacija
        #anim.plot_car_RT(px, py, psi, 1, delta,i*0.1)
        #time.sleep(0.1)
        
        simX[i] = x_ref[i]
        simY[i] = y_ref[i]
        simV[i] = 1
        simP[i] = psi
        simD[i]= delta    
    #anim.close()

    #test for animation
    anim.animate(simX, simY,simP,simV,simD,t_ref)


    
