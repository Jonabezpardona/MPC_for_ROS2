'''
The values either taken from the Gazebo model or aproximated based on other research. 
in lateral_tire_model, there is a plot of the parameters for dynamic model  
If used on the car, DOUBLE CHECK.
'''

import numpy as np
import matplotlib.pyplot as plt

class VehicleParams():
    def __init__(self):
        self.g = 9.81
        self.mi = 0.9

        #air charachteristics for Bosch car
        self.ro = 1.2
        self.Cz = 0.32
        self.Az=  0.021

    # in order to use this kind of the model, you need to know the model of the motor, which is impossible to do from Gazebo
    # however it can be done on the physical car. This requires to get rid of the PID working on the motor
    def RaceCar43(self):
        self.lf, self.lr = 0.0308, 0.0305 # distance from CoG to front and rear wheels
        self.L = self.lr + self.lf # wheel base

        self.m, self.I_z = 0.0467, 5.6919 *1e-5
        self.c1,self.c2 = 0.5, 17.06 # geometrical values-lr/l,  1/l

        self.Bcf, self.Ccf, self.Dcf = 3.47, 0.1021, 5.003  
        self.Bcr, self.Ccr, self.Dcr = 3.173, 0.01921, 19.01 

        self.Cm1, self.Cm2 = 12, 2.17
        self.cr0, self.cr2, self.cr3 = 0.01, 0.006, 5
        


    def BoschCar(self):
        # values are based on the car model from Gazebo 
        self.lf,self.lr = 0.13, 0.13

        self.L = self.lf+self.lr
        self.m = 1.415

        self.h = 0.03
        self.I_z = 0.17423

        # values are aproximated, and visually represented in Pacejka Tire Model
        # FRONT TIRE
        self.Bcf, self.Ccf, self.Dcf = 0.425, 1.3, 6.246 #0.23, 1.57, 6.94

        # REAR TIRE
        self.Bcr, self.Ccr, self.Dcr = 0.425, 1.3, 6.246


if __name__ == "__main__":
    vp = VehicleParams()
    vp.BoschCar()  

    print(vp.m)   
    print(vp.lf) 
    

