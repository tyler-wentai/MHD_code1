import numpy as np
import time 
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu 

# This code solves the full set of ideal MHD equations using the semi-implicit
# method. 
# Code Version: Python 3.X
# Author:       Tyler E. Bagwell (2020)

# INITIALIZE --------
mxp = int(135)  # total number of points in x
myp = int(135)  # total number of points in y

dx  = np.zeros((mxp),   dtype=np.double, order='C')
xx  = np.zeros((mxp),   dtype=np.double, order='C')
xxx = np.zeros((mxp+1), dtype=np.double, order='C')     # for plotting ONLY

dy  = np.zeros((myp),   dtype=np.double, order='C')
yy  = np.zeros((myp),   dtype=np.double, order='C')
yyy = np.zeros((myp+1), dtype=np.double, order='C')     # for plotting ONLY

    # Derivative coefficients: ( index, derv. degree, three coefs (a,b,c))
dxcCOEF = np.zeros((mxp,2,3), dtype=np.double, order='C')
dxpCOEF = np.zeros((mxp,2,3), dtype=np.double, order='C')
dxmCOEF = np.zeros((mxp,2,3), dtype=np.double, order='C')

dycCOEF = np.zeros((myp,2,3), dtype=np.double, order='C')
dypCOEF = np.zeros((myp,2,3), dtype=np.double, order='C')
dymCOEF = np.zeros((myp,2,3), dtype=np.double, order='C')

ay1, by1, cy1, dy1 = 0., 0., 0., 0.
ay2, by2, cy2, dy2 = 0., 0., 0., 0.

    # The MHD variables are labeled 0 through 6 and stored in x:
    #   0:rho; 1:vx; 2:vy; 3:vz; 4:psi; 5:Bz; 6:pressure
    #   cur: current density vec(J)
x   = np.zeros((mxp,myp,7), dtype=np.double, order='C')
cur = np.zeros((mxp,myp,3), dtype=np.double, order='C')

bx  = np.zeros((mxp,myp), dtype=np.double)
by  = np.zeros((mxp,myp), dtype=np.double)

rhoINIT = np.zeros((mxp,myp), dtype=np.double)
psiINIT = np.zeros((mxp,myp), dtype=np.double)
preINIT = np.zeros((mxp,myp), dtype=np.double)
velINIT = np.zeros((mxp,myp,3), dtype=np.double)
bINIT   = np.zeros((mxp,myp,3), dtype=np.double)
curINIT = np.zeros((mxp,myp,3), dtype=np.double)


s2COEF  = np.zeros((mxp,myp), dtype=np.double)          # Smoothing strength profile for viscCOEF2


# INPUT and NORMALIZED VALUES : -------------------------
nstep, nstep_end = 0, 20001
timeSIM     = 0.            # simulated time
shearMAX    = 0.

visc1LOG    = True 
visc2LOG    = False 
gravLOG     = True
resiLOG     = False
    
viscCOEF1   = 1./100.       # coef of viscosity for general domain smoothing
viscCOEF2   = 1./15.        # coef of viscosity for specific smoothing near the origin
resiCOEF    = 1./1000.      # coef of resistivity 
    
vzMAX       = 1.0e-2        # max z-direction shear speed ratioed to normalized v
tACC        = 50.           # time required for shear to accelerate to max speed vzMAX
    
tempINIT    = 1.000e+0      # temperature (currently assumes system is isothermic)
adbtCONS    = 1.000e+0      # adibatic constant
gasR        = 1.624e-3      # gas constant
sunR        = 2.320e+1      # solar radius
surfGRAV    = 8.027e-4      # surface gravity 

cx, cy, cz, cs = 0., 0., 0., 0.

unifrmSET = False           # Uniform grid (TRUE) or nonuniform grid (FALSE)
halfxdSET = False           # Model half of the x-domain (TRUE) or entire x-domain (FALSE)

xmin, xmax = -5., 5.
ymin, ymax = 0., 10.0
dx_min, dy_min = 0., 0. 

#cspeedFILE      = open("cspeedFILE.txt", 'w')
#vxPRDCT_FILE    = open("vxPRDCT_FILE.txt", 'w')
#vxADVCD_FILE    = open("vxADVCD_FILE.txt", 'w')
cA_FILE    = open("cA_FILE.txt", 'w')
    
### ------------- MAIN FUNCTION ------------- ###
def main():
    global nstep, timeSIM, shearMAX

    set_grid()
    set_coef()
    initial()
    
    cfl             = 0.25
    velMAX          = calc_velMAX('1',cfl)
    dt              = cfl*min(dx_min,dy_min)/velMAX
    
    print("dx_min: {a}, dy_min: {b}, dt: {c}" .format(a=dx_min,b=dy_min,c=dt))
    
    
    # Init plotting
    diagnostic_plotter(cfl)
    
    #    jzFILE      = open("jzfile.txt", 'w')
    shearFILE   = open("shearfile.txt", 'w')
    shearFILE.write("%s\n" %shearMAX)
    
    # Main Loop -------------------
    for n in range(nstep_end+1):
    
        time_start = time.time()
        stepon(dt, timeSIM)
        calc_velMAX('2',cfl)
        print("%.5f secs for step %i" %(time.time() - time_start,nstep))
                
        nstep       += 1
        timeSIM     += dt
        if (timeSIM < tACC):
            shearMAX += (timeSIM/tACC)*vzMAX*dt
        else:
            shearMAX += vzMAX*dt
            
        ### Diagnostics ###
        #        jzFILE.write("%s\n" %np.abs(np.min(cur[:,:,2])))
        shearFILE.write("%s\n" %shearMAX)
        
        if (n%1000==0):
            print("Sim time: %.5f; Sim shear: %.5f" %(timeSIM,shearMAX))
            diagnostic_plotter(cfl)

    # End Main Loop ---------------
    
    #    jzFILE.close()
    #    shearFILE.close()
    

### ---------------------------------------- ###

#
def set_grid():
    global dx_min, dy_min
    if unifrmSET==True:
        dxi = np.float((xmax-xmin)/float(mxp-1))
        dyi = np.float((ymax-ymin)/float(myp-1))
                
        dx_min, dy_min = dxi, dyi

        for i in range(mxp):
            dx[i], xx[i] = dxi, xmin + i*dxi
        for j in range(myp):
            dy[j], yy[j] = dyi, ymin + j*dyi
            
        for i in range(mxp+1):
            xxx[i] = xmin + i*dxi
        for j in range(myp+1):
            yyy[j] = ymin + j*dyi
            
    else:
        dx_min = 1./11.
        dx_max = 7.50*dx_min
        dy_min = 1./11.
        dy_max = 7.50*dy_min

        
        print( "Min dx: %f;  Max dx: %f" %(dx_min,dx_max))
        print( "Min dy: %f;  Max dy: %f" %(dy_min,dy_max))
        
        xSPREAD, ySPREAD = 13., 20.
        
        mxpHALF = int(((mxp-1)/2)+1)
        
        # x:
        def dx_calc(i):
            return 0.5*(dx_max+dx_min) + 0.5*(dx_max-dx_min)*np.tanh((i-int(mxpHALF/2))/xSPREAD)
            
#        for i in range(mxp):
#            dx[i] = dx_calc(i)
#            xx[i] = xmin + sum(dx[0:i])
#            
#        for i in range(mxp+1):
#            if (i==myp):
#                xxx[i] = xxx[i-1]+dx[i-1]
#            else:
#                xxx[i] = xmin + sum(dx[0:i])

        xx_help = np.zeros(mxpHALF)
        dx_help = np.zeros(mxpHALF)
        for i in range(0,mxpHALF):
            dx_help[i] = dx_calc(i)
            if (i==0):  xx_help[i] = 0.
            else:       xx_help[i] = xx_help[i-1] + dx_help[i]
                
        for i in range(mxpHALF-1):
            xx[i] = - xx_help[-(i+1)]
                
        for i in range(mxpHALF-1,mxp):
            xx[i] = + xx_help[i-mxpHALF+1]
                

                
        for i in range(mxp+1):
            if (i==mxp):
                xxx[i] = xxx[i-1] + dx_help[-1]
            else:
                xxx[i] = xx[i]

        # y:
        def dy_calc(j):
            return 0.5*(dy_max+dy_min) + 0.5*(dy_max-dy_min)*np.tanh((j-int(myp/2))/ySPREAD)
            
        for j in range(myp):
            dy[j] = dy_calc(j)
            yy[j] = ymin + sum(dy[0:j])
            
        for j in range(myp+1):
            if (j==myp):
                yyy[j] = yyy[j-1]+dy[j-1]
            else:
                yyy[j] = ymin + sum(dy[0:j])
            
            
    lx = [i for i in range(mxp)]
    ly = [i for i in range(myp)]

    #plt.plot(lz,dz)
    #plt.xlim(0,mzp)
    #plt.ylim(dz_min,dz_max*(1.02))
    #plt.show()

    plt.plot(lx,xx)
#    plt.plot(lx,dx)
    plt.xlim(0,mxp)
    #plt.ylim(xmin,xmax*(1.02))
    plt.show()
            
    
#
def set_coef():
    global ay1, ay2, by1, by2, cy1, cy2, dy1, dy2
    # Three-point differencing is utilized which gives 2nd-order accuracy in a 
    # uniform gridding. 
    
    ### ------ CENTERED DIFFERENCING ------ ###
    ### ----- a:x_i+1, b:x_i, c:x_i-1 ----- ###
    
    for i in range(0,mxp-1):
        if (i==0):
            dx1 = xx[i] + xx[i+1]
            dx2 = xx[i+1] - xx[i]
        else:
            dx1 = xx[i] - xx[i-1]
            dx2 = xx[i+1] - xx[i]
            
        denom = dx1*dx2*(dx1+dx2)
        
        # coefs 1st derivatives in x:
        dxcCOEF[i,0,0] = +dx1**2/denom
        dxcCOEF[i,0,1] = +(dx2**2 - dx1**2)/denom
        dxcCOEF[i,0,2] = -dx2**2/denom
        # coefs 2nd derivatives in x:
        dxcCOEF[i,1,0] = +2.0*dx1/denom
        dxcCOEF[i,1,1] = -2.0*(dx1 + dx2)/denom
        dxcCOEF[i,1,2] = +2.0*dx2/denom
        
    for j in range(1,myp-1):
        dy1 = yy[j] - yy[j-1]
        dy2 = yy[j+1] - yy[j]
        
        denom = dy1*dy2*(dy1+dy2)
        
        # coefs 1st derivatives in y:
        dycCOEF[j,0,0] = +dy1**2/denom
        dycCOEF[j,0,1] = +(dy2**2 - dy1**2)/denom
        dycCOEF[j,0,2] = -dy2**2/denom
        # coefs 2nd derivatives in y:
        dycCOEF[j,1,0] = +2.0*dy1/denom
        dycCOEF[j,1,1] = -2.0*(dy1 + dy2)/denom
        dycCOEF[j,1,2] = +2.0*dy2/denom
    
    ### --- FORWARD-BIASED DIFFERENCING --- ###
    ### ----- a:x_i+2, b:x_i+1, c:x_i ----- ###
    for i in range(0,mxp-2):
        dx1 = xx[i+1] - xx[i]
        dx2 = xx[i+2] - xx[i+1]
        
        denom = dx1*dx2*(dx1+dx2)
        
        # coefs 1st derivatives in x:
        dxpCOEF[i,0,0] = -dx1**2/denom
        dxpCOEF[i,0,1] = +(dx1+dx2)**2/denom
        dxpCOEF[i,0,2] = -(dx2**2 + 2.0*dx1*dx2)/denom
        # coefs 2nd derivatives in x:
        dxpCOEF[i,1,0] = +2.0*dx1/denom
        dxpCOEF[i,1,1] = -2.0*(dx1 + dx2)/denom
        dxpCOEF[i,1,2] = +2.0*dx2/denom
        
    for j in range(0,myp-2):
        dy1 = yy[j+1] - yy[j]
        dy2 = yy[j+2] - yy[j+1]
        
        denom = dy1*dy2*(dy1+dy2)
        
        # coefs 1st derivatives in y:
        dypCOEF[j,0,0] = -dy1**2/denom
        dypCOEF[j,0,1] = +(dy1+dy2)**2/denom
        dypCOEF[j,0,2] = -(dy2**2 + 2.0*dy1*dy2)/denom
        # coefs 2nd derivatives in y:
        dypCOEF[j,1,0] = +2.0*dy1/denom
        dypCOEF[j,1,1] = -2.0*(dy1 + dy2)/denom
        dypCOEF[j,1,2] = +2.0*dy2/denom
        
    ### --- BACKWARD-BIASED DIFFERENCING --- ###
    ### ----- a:x_i, b:x_i-1, c:x_i-2 ------ ###
    for i in range(2,mxp):
        dx1 = xx[i-1] - xx[i-2]
        dx2 = xx[i] - xx[i-1]
        
        denom = dx1*dx2*(dx1+dx2)
        
        # coefs 1st derivatives in x:
        dxmCOEF[i,0,0] = +(dx1**2 + 2.0*dx1*dx2)/denom
        dxmCOEF[i,0,1] = -(dx1+dx2)**2/denom
        dxmCOEF[i,0,2] = +dx2**2/denom
        # coefs 2nd derivatives in x:
        dxmCOEF[i,1,0] = +2.0*dx1/denom
        dxmCOEF[i,1,1] = -2.0*(dx1+dx2)/denom
        dxmCOEF[i,1,2] = +2.0*dx2/denom
        
    for j in range(2,myp):
        dy1 = yy[j-1] - yy[j-2]
        dy2 = yy[j] - yy[j-1]
        
        denom = dy1*dy2*(dy1+dy2)
        
        # coefs 1st derivatives in y:
        dymCOEF[j,0,0] = +(dy1**2 + 2.0*dy1*dy2)/denom
        dymCOEF[j,0,1] = -(dy1+dy2)**2/denom
        dymCOEF[j,0,2] = +dy2**2/denom
        # coefs 2nd derivatives in y:
        dymCOEF[j,1,0] = +2.0*dy1/denom
        dymCOEF[j,1,1] = -2.0*(dy1+dy2)/denom
        dymCOEF[j,1,2] = +2.0*dy2/denom
        
    ### ------- SMOOTHING PROFILE 2 ------- ###
    ### ----------------------------------- ###
    for i in range(0,mxp):
        for j in range(0,myp):
            s2COEF[i,j] = 1.0 - 1.0*np.tanh(yy[j]/0.5)
            #s2COEF[i,j] = 1.0 - 1.0*np.tanh(np.sqrt(xx[i]**2 + yy[j]**2)/1.0)
            
    plt.pcolormesh(xxx,yyy,s2COEF.T)
    cb = plt.colorbar()
    plt.show()

    ### ------- 4-POINT DIFFERENCING ------- ###
    ### ------------------------------------ ###
    
    a = yy[1] - yy[0]
    b = yy[2] - yy[1]
    c = yy[3] - yy[1]
    #
    ay1 = a*b*(a+b)/(a**2*b*c + a*b**2*c - a**2*c**2 + b**2*c**2 - a*c**3 - b*c**3)
    by1 = a*c*(a+c)/(a**2*b*c + a*b*c**2 - a**2*b**2 - a*b**3 - b**3*c + b**2*c**2)
    cy1 = (a**3*b**2 + a**2*b**3 - a**3*c**2 - b**3*c**2 - a**2*c**3 + b**2*c**3)    \
         /(a*b*c*( -a**2*b - a*b**2 + a**2*c - b**2*c + a*c**2 + b*c**2 ))
    dy1 = b*c*(b-c)/(-a**3*b - a**2*b**2 + a**3*c - a*b**2*c + a**2*c**2 + a*b*c**2)
    #
    ay2 = 2.0*(a**2-b**2)/( -a**2*b*c - a*b**2*c + a**2*c**2 - b**2*c**2 + a*c**3 + b*c**3 )
    by2 = 2.0*(c**2-a**2)/( -a**2*b**2 - a*b**3 + a**2*b*c - b**3*c + a*b*c**2 + b**2*c**2 )
    cy2 = 2.0*(-a**3*b + a*b**3 + a**3*c + b**3*c - a*c**3 - b*c**3) \
            /(a*b*c*( -a**2*b - a*b**2 + a**2*c - b**2*c + a*c**2 + b*c**2 ))
    dy2 = 2.0*(c**2-b**2)/( -a**3*b - a**2*b**2 + a**3*c - a*b**2*c + a**2*c**2 + a*b*c**2 )

#
def dfdxc(order,index,fp1,f0,fm1):
    global dxcCOEF
    if      (order=='1'): coef1, coef2, coef3 = dxcCOEF[index,0,:]
    elif    (order=='2'): coef1, coef2, coef3 = dxcCOEF[index,1,:]
    else:   raise ValueError('Derivative order error: should be "1" or "2".')
    return (coef1*fp1 + coef2*f0 + coef3*fm1)
    
def dfdyc(order,index,fp1,f0,fm1):
    global dycCOEF
    if      (order=='1'): coef1, coef2, coef3 = dycCOEF[index,0,:]
    elif    (order=='2'): coef1, coef2, coef3 = dycCOEF[index,1,:]
    else:   raise ValueError('Derivative order error: should be "1" or "2".')
    return (coef1*fp1 + coef2*f0 + coef3*fm1)
    
def dfdxp(order,index,fp2,fp1,f0):
    global dxpCOEF
    if      (order=='1'): coef1, coef2, coef3 = dxpCOEF[index,0,:]
    elif    (order=='2'): coef1, coef2, coef3 = dxpCOEF[index,1,:]
    else:   raise ValueError('Derivative order error: should be "1" or "2".')
    return (coef1*fp2 + coef2*fp1 + coef3*f0)
    
def dfdyp(order,index,fp2,fp1,f0):
    global dypCOEF
    if      (order=='1'): coef1, coef2, coef3 = dypCOEF[index,0,:]
    elif    (order=='2'): coef1, coef2, coef3 = dypCOEF[index,1,:]
    else:   raise ValueError('Derivative order error: should be "1" or "2".')
    return (coef1*fp2 + coef2*fp1 + coef3*f0)
    
def dfdxm(order,index,f0,fm1,fm2):
    global dxmCOEF
    if      (order=='1'): coef1, coef2, coef3 = dxmCOEF[index,0,:]
    elif    (order=='2'): coef1, coef2, coef3 = dxmCOEF[index,1,:]
    else:   raise ValueError('Derivative order error: should be "1" or "2".')
    return (coef1*f0 + coef2*fm1 + coef3*fm2)
    
def dfdym(order,index,f0,fm1,fm2):
    global dymCOEF
    if      (order=='1'): coef1, coef2, coef3 = dymCOEF[index,0,:]
    elif    (order=='2'): coef1, coef2, coef3 = dymCOEF[index,1,:]
    else:   raise ValueError('Derivative order error: should be "1" or "2".')
    return (coef1*f0 + coef2*fm1 + coef3*fm2)
    
def dfdy_fb(order,index,fp2,fp1,f0,fm1):
    global dymCOEF
    if (index==1):
        if      (order=='1'): coef1, coef2, coef3, coef4 = ay1, by1, cy1, dy1
        elif    (order=='2'): coef1, coef2, coef3, coef4 = ay2, by2, cy2, dy2
        else:   raise ValueError('Derivative order error: should be "1" or "2".')
        return (coef1*fp2 + coef2*fp1 + coef3*f0 + coef4*fm1)
    else:
        raise ValueError('4-Point Derivative Index is Invalid.')
    
    
#    
def initial():
    # Set the initial system. Currently, the initial MHD system will begin in a
    # typical hydrostatic equilibrium with an embedded, force-free magnetic field.
    
    hydrostatCOEF = surfGRAV*sunR/gasR/tempINIT
    rhoBASE = 1.000e+0
    
#    # Euler-like method to numerically calculate initial mass density profile (rho):
#    rho_profile = np.zeros((myp), dtype=np.double)
#    
#    rho_profile[0] = rhoBASE*np.exp(-hydrostatCOEF*yy[0]/(sunR + yy[0])) # True Solution at y0
#    rho_profile[1] = rhoBASE*np.exp(-hydrostatCOEF*yy[1]/(sunR + yy[1])) # True Solution at y1
#    for j in range(1,myp-1):
#        grav = surfGRAV*sunR**2/(sunR+yy[j])**2
#
#        rho_profile[j+1] = - (  rho_profile[j+0]*dycCOEF[j,0,1]       \
#                             +  rho_profile[j-1]*dycCOEF[j,0,2]       \
#                             + (rho_profile[j+0]*grav/gasR/tempINIT) )   \
#                             /(dycCOEF[j,0,0])                

    # Set the initial values.
    for i in range(mxp):
        for j in range(myp):
        
            rhoINIT[i,j] = rhoBASE*np.exp(-hydrostatCOEF*yy[j]/(sunR + yy[j]))
    
            velINIT[i,j,0] = 0.0      # vx
            velINIT[i,j,1] = 0.0      # vy
            velINIT[i,j,2] = 0.0      # vz
            
            fieldDEN = xx[i]**2 + yy[j]**2 + 2.0*yy[j]*np.sqrt(3) + 3.0
    
            psiINIT[i,j] = (8.0/np.sqrt(3.0))*(yy[j] + np.sqrt(3.0))/fieldDEN
    
            bINIT[i,j,0] = (8.0/np.sqrt(3.0))*(-xx[i]**2 + yy[j]**2 + 2.0*yy[j]*np.sqrt(3.0) + 3.0)/fieldDEN**2
            bINIT[i,j,1] = (-16.0/np.sqrt(3.0))*((yy[j]+np.sqrt(3.0))*xx[i])/fieldDEN**2
            bINIT[i,j,2] = 0.0      # bz
    
            preINIT[i,j] = rhoINIT[i,j] * gasR * tempINIT
    
            curINIT[i,j,0] = 0.0    # jx
            curINIT[i,j,1] = 0.0    # jy
            curINIT[i,j,2] = 0.0    # jz
            
    # Load initial values into x variable array. 
    for i in range(mxp):
        for j in range(myp):
        
            x[i,j,0] = rhoINIT[i,j]
            
            x[i,j,1] = velINIT[i,j,0]
            x[i,j,2] = velINIT[i,j,1]
            x[i,j,3] = velINIT[i,j,2]
            
            x[i,j,4] = psiINIT[i,j]
            
            x[i,j,5] = bINIT[i,j,2]
            
            x[i,j,6] = preINIT[i,j]
            
            bx[i,j] = bINIT[i,j,0]
            by[i,j] = bINIT[i,j,1]
            
            cur[i,j,0] = curINIT[i,j,0]
            cur[i,j,1] = curINIT[i,j,1]
            cur[i,j,2] = curINIT[i,j,2]
            

    alfvenPROFILE = np.zeros((mxp,myp))
    for i in range(mxp):
        for j in range(myp):
            alfvenPROFILE[i,j] = np.sqrt(bINIT[i,j,0]**2 + bINIT[i,j,1]**2)/np.sqrt(rhoINIT[i,j])
    

    f_true  = np.zeros((myp))
    for j in range(myp):
        f_true[j] = 1.0*np.exp(-hydrostatCOEF*yy[j]/(sunR + yy[j]))
    
#    plt.plot(yy[1:myp-1],error[1:myp-1])
    plt.plot(yy,rhoINIT[0,:],color='r')
    plt.plot(yy,f_true,color='k',linestyle=':')
    plt.show()
    
    plt.pcolormesh(xxx,yyy,alfvenPROFILE.T, cmap='brg')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    cb = plt.colorbar()
    plt.title("Initial Alfven Speed Profile")
    plt.show()
    
          
#
def calc_velMAX(num_IN,cfl):
    # Calculates the current maximum Alfven speed over entire computational domain.
    
    cA_MAX = 0.
    for i in range(mxp):
        for j in range(myp):
            cA_help = np.sqrt(bx[i,j]**2 + by[i,j]**2)/np.sqrt(x[i,j,0])
            
            if (cA_help > cA_MAX): cA_MAX = cA_help
            
    cLIGHT = 93.75
    if (np.abs(cA_MAX)>cLIGHT):
        diagnostic_plotter(cfl)
        print("Sim time: %.5f; Sim shear: %.5f" %(timeSIM,shearMAX))
        raise ValueError('A characteristic MHD speed exceeds speed of light.')
        
    print(cA_MAX)
    cA_FILE.write("%s\n" %cA_MAX)
        
    if (num_IN=='1'):
        return cA_MAX
    else:
        pass
    
    
            
#
def stepon(dt, timeSIM):
    # The governing equations are solved by means of a three-point center differencing and
    # a 2nd-order Runge-Kutta integration. 
    
    # xOLD:     holds the variable values at the old time t_n.
    # xRIGHT:   holds the right side values of the governing equations. 
    # x:        holds the up-to-date values always.

    
    xOLD    = x.copy()
    xRIGHT  = np.zeros((mxp,myp,7), dtype=np.double)
    
    # 1st Step:
    calcRIGHT_INN(xRIGHT)
    calcRIGHT_OUT(timeSIM, xRIGHT)
    for s in range(7):
        for i in range(mxp):
            for j in range(myp):
                x[i,j,s] = xOLD[i,j,s] + 0.5*dt*xRIGHT[i,j,s]
    calc_bxby(x[:,:,4])
    calc_cur( x[:,:,4], x[:,:,5])
    
    # 2nd Step:
    calcRIGHT_INN(xRIGHT)
    calcRIGHT_OUT(timeSIM+0.5*dt, xRIGHT)
    for s in range(7):
        for i in range(mxp):
            for j in range(myp):
                x[i,j,s] = xOLD[i,j,s] + 1.0*dt*xRIGHT[i,j,s]
    calc_bxby(x[:,:,4])
    calc_cur( x[:,:,4], x[:,:,5])
    


# 
def calcRIGHT_INN(xRIGHT):
    # Calculates the right-hand side of the governing equations for all inner points.
    
    for i in range(1,mxp-1):
        for j in range(1,myp-1):
        
            # 0: rho
            fxp1 = x[i+1,j,0]*x[i+1,j,1]
            fx0  = x[i+0,j,0]*x[i+0,j,1]
            fxm1 = x[i-1,j,0]*x[i-1,j,1]
            
            fyp1 = x[i,j+1,0]*x[i,j+1,2]
            fy0  = x[i,j+0,0]*x[i,j+0,2]
            fym1 = x[i,j-1,0]*x[i,j-1,2]
            
            xRIGHT[i,j,0] = - ( dfdxc('1',i,fxp1,fx0,fxm1)  \
                              + dfdyc('1',j,fyp1,fy0,fym1)  )
                              
            # 1: vx
            fxp1 = x[i+1,j,1]
            fx0  = x[i+0,j,1]
            fxm1 = x[i-1,j,1]
            
            fyp1 = x[i,j+1,1]
            fy0  = x[i,j+0,1]
            fym1 = x[i,j-1,1]

            xRIGHT[i,j,1] = - ( x[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)                             \
                              + x[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                             \
                              + (dfdxc('1',i,x[i+1,j,6],x[i+0,j,6],x[i-1,j,6])/x[i,j,0])        \
                              + ((cur[i,j,2]*by[i,j] - cur[i,j,1]*x[i,j,5])/x[i,j,0])           )
                              
            # 2: vy
            fxp1 = x[i+1,j,2]
            fx0  = x[i+0,j,2]
            fxm1 = x[i-1,j,2]

            fyp1 = x[i,j+1,2]
            fy0  = x[i,j+0,2]
            fym1 = x[i,j-1,2]
            
            xRIGHT[i,j,2] = - ( x[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)                             \
                              + x[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                             \
                              + ((cur[i,j,0]*x[i,j,5] - cur[i,j,2]*bx[i,j])/x[i,j,0])           )
                              
            if (gravLOG == True):
                delRHO_p1 = x[i,j+1,0] - rhoINIT[i,j+1]
                delRHO_00 = x[i,j+0,0] - rhoINIT[i,j+0]
                delRHO_m1 = x[i,j-1,0] - rhoINIT[i,j-1]
                
                xRIGHT[i,j,2] -= (gasR*tempINIT*dfdyc('1',j,delRHO_p1,delRHO_00,delRHO_m1)      \
                                          + (delRHO_00*surfGRAV*(sunR**2)/(sunR + yy[j])**2)  ) \
                                          /x[i,j,0]
            else:
                xRIGHT[i,j,2] -= dfdyc('1',j,x[i,j+1,6],x[i,j+0,6],x[i,j-1,6])/x[i,j,0]
                
            # 3: vz
            fxp1 = x[i+1,j,3]
            fx0  = x[i+0,j,3]
            fxm1 = x[i-1,j,3]

            fyp1 = x[i,j+1,3]
            fy0  = x[i,j+0,3]
            fym1 = x[i,j-1,3]
            
            xRIGHT[i,j,3] = - ( x[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)                    \
                              + x[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                    \
                              + ((cur[i,j,1]*bx[i,j] - cur[i,j,0]*by[i,j])/x[i,j,0])   )
                              
            # 4: psi
            fxp1 = x[i+1,j,4]
            fx0  = x[i+0,j,4]
            fxm1 = x[i-1,j,4]

            fyp1 = x[i,j+1,4]
            fy0  = x[i,j+0,4]
            fym1 = x[i,j-1,4]
            
            xRIGHT[i,j,4] = - ( x[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)  \
                              + x[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)  )

            # 5: Bz
            fxp1 = x[i+1,j,3]*bx[i+1,j] - x[i+1,j,1]*x[i+1,j,5]
            fx0  = x[i+0,j,3]*bx[i+0,j] - x[i+0,j,1]*x[i+0,j,5]
            fxm1 = x[i-1,j,3]*bx[i-1,j] - x[i-1,j,1]*x[i-1,j,5]

            fyp1 = x[i,j+1,3]*by[i,j+1] - x[i,j+1,2]*x[i,j+1,5]
            fy0  = x[i,j+0,3]*by[i,j+0] - x[i,j+0,2]*x[i,j+0,5]
            fym1 = x[i,j-1,3]*by[i,j-1] - x[i,j-1,2]*x[i,j-1,5]
            
            xRIGHT[i,j,5] = + ( dfdxc('1',i,fxp1,fx0,fxm1)  \
                              + dfdyc('1',j,fyp1,fy0,fym1)  )
  
            # 6: pressure
            xRIGHT[i,j,6] = xRIGHT[i,j,0]*gasR*tempINIT
            
            
    # ----- Apply smoothing to xRIGHT velocity components -----
    if (visc1LOG==True):
        for s in range(1,4):
            smoothing_1(x[:,:,s],xRIGHT[:,:,s])


#
def smoothing_1(v_IN, vRIGHT):
    # Applies smoothing to the xRIGHT velocity components. 
    
    if (halfxdSET == False):
        for i in range(1,mxp-1):
            for j in range(1,myp-1):
        
                fxp1 = v_IN[i+1,j]
                fx0  = v_IN[i+0,j]
                fxm1 = v_IN[i-1,j]

                fyp1 = v_IN[i,j+1]
                fy0  = v_IN[i,j+0]
                fym1 = v_IN[i,j-1]
            
                vRIGHT[i,j] += viscCOEF1*( dfdxc('2',i,fxp1,fx0,fxm1)    \
                                         + dfdyc('2',j,fyp1,fy0,fym1)    )
                                         
    else:
        pass


#
def calcRIGHT_OUT(time_IN, xRIGHT):
    # Calculates the right-hand side of the governing equations for all the outer points. 

    
    # ----- (1) Left-Hand Side Boundary -----
    if (halfxdSET == False): 
        for j in range(1,myp-1):
    
            # 0: rho
            xRIGHT[0,j,0] = - ( (dxpCOEF[0,0,0]*xRIGHT[2,j,0]   \
                               + dxpCOEF[0,0,1]*xRIGHT[1,j,0])  \
                              /dxpCOEF[0,0,2] )
                          
            # 1: vx 
            xRIGHT[0,j,1] = - ( (dxpCOEF[0,0,0]*xRIGHT[2,j,1]   \
                               + dxpCOEF[0,0,1]*xRIGHT[1,j,1])  \
                              /dxpCOEF[0,0,2] )
        
            # 2: vy 
            xRIGHT[0,j,2] = - ( (dxpCOEF[0,0,0]*xRIGHT[2,j,2]   \
                               + dxpCOEF[0,0,1]*xRIGHT[1,j,2])  \
                              /dxpCOEF[0,0,2] )

            # 3: vz 
            xRIGHT[0,j,3] = - ( (dxpCOEF[0,0,0]*xRIGHT[2,j,3]   \
                               + dxpCOEF[0,0,1]*xRIGHT[1,j,3])  \
                              /dxpCOEF[0,0,2] )

            # 4: psi 
            fyp1 = x[0,j+1,4]
            fy0  = x[0,j+0,4]
            fym1 = x[0,j-1,4]
        
            xRIGHT[0,j,4] = - x[0,j,2]*dfdyc('1',j,fyp1,fy0,fym1)
        
            if (x[0,j,1] < 0.):
                fxp2 = x[2,j,4]
                fxp1 = x[1,j,4]
                fx0  = x[0,j,4]
            
                xRIGHT[0,j,4] -= x[0,j,1]*dfdxp('1',0,fxp2,fxp1,fx0)
            
            # 5: Bz
            xRIGHT[0,j,5] = - ( (dxpCOEF[0,0,0]*xRIGHT[2,j,5]   \
                               + dxpCOEF[0,0,1]*xRIGHT[1,j,5])  \
                              /dxpCOEF[0,0,2] )
                          
            # 6: Pressure
            xRIGHT[0,j,6] = xRIGHT[0,j,0]*gasR*tempINIT
            
    else:
        pass

    # ----- (2) Top Side Boundary -----
    for i in range(0,mxp-1):
    
        # 0: rho [NEED TO CHANGE!!! NOT APPROPRAITE FOR STRATIEFIED ATMOSPHERE!]
        xRIGHT[i,myp-1,0] = - ( (dymCOEF[myp-1,0,1]*xRIGHT[i,myp-2,0]   \
                               + dymCOEF[myp-1,0,2]*xRIGHT[i,myp-3,0])  \
                                /dymCOEF[myp-1,0,0] )
        
        # 1: vx
        xRIGHT[i,myp-1,1] = - ( (dymCOEF[myp-1,0,1]*xRIGHT[i,myp-2,1]   \
                               + dymCOEF[myp-1,0,2]*xRIGHT[i,myp-3,1])  \
                                /dymCOEF[myp-1,0,0] )

        # 2: vy
        xRIGHT[i,myp-1,2] = - ( (dymCOEF[myp-1,0,1]*xRIGHT[i,myp-2,2]   \
                               + dymCOEF[myp-1,0,2]*xRIGHT[i,myp-3,2])  \
                                /dymCOEF[myp-1,0,0] )
        
        # 3: vz
        xRIGHT[i,myp-1,3] = - ( (dymCOEF[myp-1,0,1]*xRIGHT[i,myp-2,3]   \
                               + dymCOEF[myp-1,0,2]*xRIGHT[i,myp-3,3])  \
                                /dymCOEF[myp-1,0,0] )

        # 4: psi
        if (i==0): 
            fxp2 = x[2,myp-1,4]
            fxp1 = x[1,myp-1,4]
            fx0  = x[0,myp-1,4]
        
            xRIGHT[i,myp-1,4] = - x[i,myp-1,1]*dfdxp('1',i,fxp2,fxp1,fx0)
            
        else:
            fxp1 = x[i+1,myp-1,4]
            fx0  = x[i+0,myp-1,4]
            fxm1 = x[i-1,myp-1,4]
        
            xRIGHT[i,myp-1,4] = - x[i,myp-1,1]*dfdxc('1',i,fxp1,fx0,fxm1)
        
        if (x[i,myp-1,2] > 0.):
            fy0  = x[i,myp-1,4]
            fym1 = x[i,myp-2,4]
            fym2 = x[i,myp-3,4]
            
            xRIGHT[i,myp-1,4] -= x[i,myp-1,2]*dfdym('1',myp-1,fy0,fym1,fym2)

        # 5: Bz
        xRIGHT[i,myp-1,5] = - ( (dymCOEF[myp-1,0,1]*xRIGHT[i,myp-2,5]   \
                               + dymCOEF[myp-1,0,2]*xRIGHT[i,myp-3,5])  \
                                /dymCOEF[myp-1,0,0] )
                                
        # 6: Pressure
        xRIGHT[i,myp-1,6] = xRIGHT[i,myp-1,0]*gasR*tempINIT
                                
                                
    # ----- (3) Right-Hand Side Boundary ----- 
    for j in range(1,myp):
    
        # 0: rho
        xRIGHT[mxp-1,j,0] = - ( (dxmCOEF[mxp-1,0,1]*xRIGHT[mxp-2,j,0]   \
                               + dxmCOEF[mxp-1,0,2]*xRIGHT[mxp-3,j,0])  \
                              /dxmCOEF[mxp-1,0,0] )
                          
        # 1: vx 
        xRIGHT[mxp-1,j,1] = - ( (dxmCOEF[mxp-1,0,1]*xRIGHT[mxp-2,j,1]   \
                               + dxmCOEF[mxp-1,0,2]*xRIGHT[mxp-3,j,1])  \
                              /dxmCOEF[mxp-1,0,0] )
        
        # 2: vy 
        xRIGHT[mxp-1,j,2] = - ( (dxmCOEF[mxp-1,0,1]*xRIGHT[mxp-2,j,2]   \
                               + dxmCOEF[mxp-1,0,2]*xRIGHT[mxp-3,j,2])  \
                              /dxmCOEF[mxp-1,0,0] )

        # 3: vz 
        xRIGHT[mxp-1,j,3] = - ( (dxmCOEF[mxp-1,0,1]*xRIGHT[mxp-2,j,3]   \
                               + dxmCOEF[mxp-1,0,2]*xRIGHT[mxp-3,j,3])  \
                              /dxmCOEF[mxp-1,0,0] )

        # 4: psi 
        if (j==myp-1):
            fy0  = x[mxp-1,j+0,4]
            fym1 = x[mxp-1,j-1,4]
            fym2 = x[mxp-1,j-2,4]
        
            xRIGHT[mxp-1,j,4] = - x[mxp-1,j,2]*dfdym('1',j,fy0,fym1,fym2)
        
        else:
            fyp1 = x[mxp-1,j+1,4]
            fy0  = x[mxp-1,j+0,4]
            fym1 = x[mxp-1,j-1,4]
        
            xRIGHT[mxp-1,j,4] = - x[mxp-1,j,2]*dfdyc('1',j,fyp1,fy0,fym1)
        
        if (x[mxp-1,j,1] > 0.):
            fx0  = x[mxp-1,j,4]
            fxm1 = x[mxp-2,j,4]
            fxm2 = x[mxp-3,j,4]
            
            xRIGHT[mxp-1,j,4] -= x[mxp-1,j,1]*dfdxm('1',mxp-1,fx0,fxm1,fxm2)
            
        # 5: Bz
        xRIGHT[mxp-1,j,5] = - ( (dxmCOEF[mxp-1,0,1]*xRIGHT[mxp-2,j,5]   \
                               + dxmCOEF[mxp-1,0,2]*xRIGHT[mxp-3,j,5])  \
                              /dxmCOEF[mxp-1,0,0] )
                          
        # 6: Pressure
        xRIGHT[mxp-1,j,6] = xRIGHT[mxp-1,j,0]*gasR*tempINIT
        
                                
    # ----- (4) Bottom Side Boundary -----
    for i in range(mxp):
    
        # 0: rho
        xRIGHT[i,0,0] = - x[i,0,0]*dfdyp('1',0,x[i,2,2],x[i,1,2],x[i,0,2])
        
        # 1: vx
        xRIGHT[i,0,1] = 0.
        
        # 2: vy
        xRIGHT[i,0,2] = 0.
        
        # 3: vz
        if (time_IN < tACC):
            vzPROFILE = vzMAX*xx[i]*np.exp(0.5*(1.0 - xx[i]**2))
            xRIGHT[i,0,3] = vzPROFILE/tACC
        else:
            xRIGHT[i,0,3] = 0.

        # 4: psi
        xRIGHT[i,0,4] = 0.
        
        # 5: Bz
        if (i==0):
            DERVxA = dfdxp('1',i,bx[2,0],bx[1,0],bx[0,0])
            DERVxB = dfdxp('1',i,x[2,0,3],x[1,0,3],x[0,0,3])
            
            DERVyA = dfdyp('1',0,by[i,2],by[i,1],by[i,0])
            DERVyB = dfdyp('1',0,x[i,2,3],x[i,1,3],x[i,0,3])
            DERVyC = dfdyp('1',0,x[i,2,2],x[i,1,2],x[i,0,2])
        
        elif (i==mxp-1):
            DERVxA = dfdxm('1',i,bx[i+0,0],bx[i-1,0],bx[i-2,0])
            DERVxB = dfdxm('1',i,x[i+0,0,3],x[i-1,0,3],x[i-2,0,3])
            
            DERVyA = dfdyp('1',0,by[i,2],by[i,1],by[i,0])
            DERVyB = dfdyp('1',0,x[i,2,3],x[i,1,3],x[i,0,3])
            DERVyC = dfdyp('1',0,x[i,2,2],x[i,1,2],x[i,0,2])
                                               
        else:
            DERVxA = dfdxc('1',i,bx[i+1,0],bx[i+0,0],bx[i-1,0])
            DERVxB = dfdxc('1',i,x[i+1,0,3],x[i+0,0,3],x[i-1,0,3])
            
            DERVyA = dfdyp('1',0,by[i,2],by[i,1],by[i,0])
            DERVyB = dfdyp('1',0,x[i,2,3],x[i,1,3],x[i,0,3])
            DERVyC = dfdyp('1',0,x[i,2,2],x[i,1,2],x[i,0,2])
            
        xRIGHT[i,0,5] = (    DERVxA*x[i,0,3]    \
                         +   DERVxB*bx[i,0]     \
                         +   DERVyA*x[i,0,3]    \
                         +   DERVyB*by[i,0]     \
                         -   DERVyC*x[i,0,5]    )

        # 6: Pressure
        xRIGHT[i,0,6] = xRIGHT[i,0,0]*gasR*tempINIT


#
def calc_bxby(psi_IN):
    # Calculates the first-order predicted values for Bx and By.
    
    w = np.zeros((mxp,myp), dtype=np.double)
    
    for i in range(mxp):
        for j in range(myp):
            w[i,j] = psi_IN[i,j] - psiINIT[i,j]
            
    # Calculate dBx:
    for i in range(mxp):
        for j in range(myp):
            if (j==0):
                bx[i,j] = - dfdyp('1',j,w[i,j+2],w[i,j+1],w[i,j+0])
            elif (j==myp-1):
                bx[i,j] = - dfdym('1',j,w[i,j+0],w[i,j-1],w[i,j-2])
            else:
                bx[i,j] = - dfdyc('1',j,w[i,j+1],w[i,j+0],w[i,j-1])
                
    # Calculate dBy:
    for i in range(mxp):
        for j in range(myp):
            if (j==0):
                by[i,j] = 0.0
            else:
                if (i==0):
                    by[i,j] = + dfdxp('1',i,w[i+2,j],w[i+1,j],w[i+0,j])
                elif (i==mxp-1):
                    by[i,j] = + dfdxm('1',i,w[i+0,j],w[i-1,j],w[i-2,j])
                else:
                    by[i,j] = + dfdxc('1',i,w[i+1,j],w[i+0,j],w[i-1,j])
                    
    # Add the initial Bx and By values to dBx and dBy:
    for i in range(mxp):
        for j in range(myp):
            bx[i,j] = bx[i,j] + bINIT[i,j,0]
            by[i,j] = by[i,j] + bINIT[i,j,1]
            

#    
def calc_cur(psi_IN, bz_IN):
    # Calculates the current values of the current density components.

    w   = np.zeros((mxp,myp), dtype=np.double)
    
    for i in range(mxp):
        for j in range(myp):
            w[i,j] = psi_IN[i,j] - psiINIT[i,j]
    
    # Calculate jx: ---------------------------------
    for i in range(mxp):
        for j in range(myp):
            if (j==0):
                cur[i,j,0] = + dfdyp('1',j,bz_IN[i,j+2],bz_IN[i,j+1],bz_IN[i,j+0])
            elif (j==myp-1):
                cur[i,j,0] = + dfdym('1',j,bz_IN[i,j+0],bz_IN[i,j-1],bz_IN[i,j-2])
            else:
                cur[i,j,0] = + dfdyc('1',j,bz_IN[i,j+1],bz_IN[i,j+0],bz_IN[i,j-1])
    
    # Calculate jy: ---------------------------------
    for i in range(mxp):
        for j in range(myp):
            if (i==0):
                cur[i,j,1] = - dfdxp('1',i,bz_IN[i+2,j],bz_IN[i+1,j],bz_IN[i+0,j])
            elif (i==mxp-1):
                cur[i,j,1] = - dfdxm('1',i,bz_IN[i+0,j],bz_IN[i-1,j],bz_IN[i-2,j])
            else:
                cur[i,j,1] = - dfdxc('1',i,bz_IN[i+1,j],bz_IN[i+0,j],bz_IN[i-1,j])
                            
    # Calculate jz: ---------------------------------
    for i in range(1,mxp-1):
        for j in range(1,myp-1):
            cur[i,j,2] =   dfdxc('2',i,w[i+1,j],w[i+0,j],w[i-1,j]) \
                         + dfdyc('2',j,w[i,j+1],w[i,j+0],w[i,j-1]) 
                                  
        # jz left-side boundary:
    i = 0
    for j in range(1,myp):
        if (j==myp-1):
            cur[i,j,2] =   dfdxp('2',i,w[i+2,j],w[i+1,j],w[i+0,j]) \
                         + dfdym('2',j,w[i,j+0],w[i,j-1],w[i,j-2])
        else:
            cur[i,j,2] =   dfdxp('2',i,w[i+2,j],w[i+1,j],w[i+0,j]) \
                         + dfdyc('2',j,w[i,j+1],w[i,j+0],w[i,j-1])
        
        # jz top boundary:
    j = myp-1
    for i in range(1,mxp):
        if (i==mxp-1):
            cur[i,j,2] =   dfdxm('2',i,w[i+0,j],w[i-1,j],w[i-2,j]) \
                         + dfdym('2',j,w[i,j+0],w[i,j-1],w[i,j-2])
        else:
            cur[i,j,2] =   dfdxc('2',i,w[i+1,j],w[i+0,j],w[i-1,j]) \
                         + dfdym('2',j,w[i,j+0],w[i,j-1],w[i,j-2])
                              
        # jz right-side boundary:
    i = mxp-1
    for j in range(1,myp-1):
        cur[i,j,2] =   dfdxm('2',i,w[i+0,j],w[i-1,j],w[i-2,j]) \
                     + dfdyc('2',j,w[i,j+1],w[i,j+0],w[i,j-1]) 
                              
        # jz bottom boundary:
    j = 0
    for i in range(0,mxp):
        cur[i,j,2] = + dfdyc('2',j,w[i,j+2],w[i,j+1],w[i,j+0])
        

#
def diagnostic_plotter(cfl):

    plt.figure(figsize=(12.00,6.00))
    f_size = 9.0
    
    w = np.zeros((mxp,myp), dtype=np.double)
    for i in range(mxp):
        for j in range(myp):
            w[i,j] = x[i,j,4] - psiINIT[i,j]
            
    dBx = np.zeros((mxp,myp), dtype=np.double)
    for i in range(mxp):
        for j in range(myp):
            dBx[i,j] = bx[i,j] - bINIT[i,j,0]
            
    dBy = np.zeros((mxp,myp), dtype=np.double)
    for i in range(mxp):
        for j in range(myp):
            dBy[i,j] = by[i,j] - bINIT[i,j,1]
            
    dRho = np.zeros((mxp,myp), dtype=np.double)
    for i in range(mxp):
        for j in range(myp):
            dRho[i,j] = x[i,j,0] - rhoINIT[i,j]
    
    # plot 1 -------------------------
    plt.subplot(231, aspect='equal')
    plt.suptitle("RKV1.3 | vis1,vis2=1/100,OFF | shear=%.5f | cfl=%.3f" %(shearMAX,cfl), fontsize=15)

    plt.pcolormesh(xxx,yyy,x[:,:,1].T, cmap='rainbow', rasterized=True)

    #plt.xlabel(r'$x}$', fontsize=f_size+0.2)
    plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Vx at Step: {a}" .format(a=nstep))


    # plot 2 -------------------------
    plt.subplot(232, aspect='equal')

    plt.pcolormesh(xxx,yyy,x[:,:,2].T, cmap='rainbow', rasterized=True)

    #plt.xlabel(r'$x$', fontsize=f_size+0.2)
    #plt.ylabel(r'$y$', fontsize=f_size+0.2)  
    cb = plt.colorbar()
    plt.title("Vy at Step: {a}" .format(a=nstep))


    # plot 3 -------------------------
    plt.subplot(233, aspect='equal')

    plt.pcolormesh(xxx,yyy,x[:,:,3].T, cmap='rainbow', rasterized=True)

#    plt.xlabel(r'$x$', fontsize=f_size+0.2)
#    plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Vz at Step: {a}" .format(a=nstep))


    # plot 4 -------------------------
    plt.subplot(234, aspect='equal')
    
    levels = 25
#    plt.contour(xx,yy,psiINIT.T, levels, colors='gray', linewidths=0.6,linestyles=':')  # no color
#    plt.contour(xx,yy,x[:,:,4].T, levels, colors='k', linewidths=0.6)                   # no color
    
    plt.pcolormesh(xxx,yyy,dRho.T, cmap='rainbow', rasterized=True)

    plt.xlabel(r'$x$', fontsize=f_size+0.2)
    plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("dRho at Step: {a}" .format(a=nstep))
    
    # plot 5 -------------------------
    plt.subplot(235, aspect='equal')
    
#    levels = 25
#    plt.contour(xx,yy,psiINIT.T, levels, colors='gray', linewidths=0.6,linestyles=':')  # no color
#    plt.contour(xx,yy,x[:,:,4].T, levels, colors='k', linewidths=0.6)                   # no color
    
    plt.pcolormesh(xxx,yyy,x[:,:,5].T, cmap='viridis', rasterized=True)

    plt.xlabel(r'$x$', fontsize=f_size+0.2)
    #plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Bz at Step: {a}" .format(a=nstep))

    
    # plot 6 -------------------------
    plt.subplot(236, aspect='equal')
    
    levels = 25
    plt.contour(xx,yy,psiINIT.T, levels, colors='gray', linewidths=0.6,linestyles=':')  # no color
    plt.contour(xx,yy,x[:,:,4].T, levels, colors='k', linewidths=0.6)                   # no color
    
    plt.pcolormesh(xxx,yyy,x[:,:,4].T, cmap='brg', rasterized=True)

    plt.xlabel(r'$x$', fontsize=f_size+0.2)
    #plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Psi and W at Step: {a}" .format(a=nstep))


    stringTEST = "RKV1.3_n" + str(nstep)
    ##plt.tight_layout(h_pad=1.75)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.10, hspace=0.20)
    plt.savefig('%s.eps' %stringTEST, format='eps', bbox_inches='tight', dpi=400)
#    plt.show()
   


# ------
main()


