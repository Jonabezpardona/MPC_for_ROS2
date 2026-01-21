'''
TRANSFORMS VALUES OF TIME DOMAIN INTO THE SPATIAL DOMAIN VIA INTERPOLATION BETWEEN POINTS
NEEDED IN ORDER TO HAVE SIMULATION AND CONTROL IN DIFFERENT DOMAINS 


instead of using just the spatial points based on the reference, that will include a small error
the position and travelled distance is calculated perciesly, even between reeference points based 
on the state projection

To determine the relative position of the vehicle with respect to a reference path
(X, Y, ψ, v), the point P where the vehicle is at the current moment is projected
onto the closest segment of the path defined by two consecutive reference points A
and B. 
A represents the reference point closest to the real position of the vehicle P, 
while B is the second closest reference point. P is somewhere in between these
two reference points. A and B are determined by searching through the array of
sampled reference positions.

Denote the segment vector as w = B - A and the vector from the point closer
to the vehicle position to P, v = P - A. The projection on the path can be
expressed in terms of the parameter

p =(wTv)/ (wTw)

and it consists of the point on the infinite line through A and B of minimum
distance from P. Notice that we consider the road segment from A to B to be
straight, since our path is sampled densely and segment can be considered nearly 
straight between two reference points.

The quantity p is a scalar that represents the position on the segment.
• p = 0 exactly at A
• p = 1 exactly at B
• 0 < p < 1 somewhere in between
• 0 > p or p > 1 outside of AB

The projected point on the path can be written in the form A + pw, thus geometrically, 
the shortest distance between the vehicle and the path line is  ∥A+ pw - P∥. The projected point 
A + pw is used to interpolate reference quantities between the two points. 

This projection ensures that (ey, eψ) are defined with respect to the nearest location on the path, 
allowing a consistent computation of tracking errors in the spatial frame.
'''

import numpy as np
def time2spatial(x,y,psi,sref,y_ref):
    xref = y_ref[0,:] 
    yref = y_ref[1, :]
    psiref = y_ref[2, :]


    idxmin=ClosestPoint(x,y,xref,yref)
    idxmin2=NextClosestPoint(x,y,xref,yref,idxmin)
    p=StateProjection(x,y,xref,yref,sref,idxmin,idxmin2)

    s0=(1-p)*sref[idxmin]+p*sref[idxmin2]
    x0=(1-p)*xref[idxmin]+p*xref[idxmin2]
    y0=(1-p)*yref[idxmin]+p*yref[idxmin2]
    psi0=(1-p)*psiref[idxmin]+p*psiref[idxmin2]

    s=s0
    ey=np.cos(psi0)*(y-y0)-np.sin(psi0)*(x-x0)
    epsi=psi-psi0

    return s,epsi,ey

def StateProjection(x,y,xref,yref,sref,idxmin,idxmin2): # linear interpolation between 2 points
    ds=abs(sref[idxmin]-sref[idxmin2]) # ds
    dxref=xref[idxmin2]-xref[idxmin]
    dyref=yref[idxmin2]-yref[idxmin]
    dx=x-xref[idxmin]
    dy=y-yref[idxmin]
    
    p =(dxref*dx+dyref*dy)/ds/ds
    return p

def ClosestPoint(x,y,xref,yref):
    min_dist=1
    idxmin=0
    for i in range(xref.size):
        dist=Eucl_dist(x,xref[i],y,yref[i])
        if dist<min_dist:
            min_dist=dist
            idxmin=i
    return idxmin

def NextClosestPoint(x,y,xref,yref,idxmin):
    prev_dist=Eucl_dist(x,xref[idxmin-1],y,yref[idxmin-1])
    next_dist=Eucl_dist(x,xref[idxmin+1],y,yref[idxmin+1])

    if(prev_dist<next_dist):
        idxmin2=idxmin-1
    else:
        idxmin2=idxmin+1

    if(idxmin2<0):
        idxmin2=xref.size-1
    elif(idxmin==xref.size):
        idxmin2=0
    
    return idxmin2

def Eucl_dist(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


