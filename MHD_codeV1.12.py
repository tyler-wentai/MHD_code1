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
mxp = int(121)  # max number of points in x
myp = int(159)  # max number of points in y

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

    # Data structures used for sparse matrix building and solving:
cxCOUNT = int(mxp+(myp-2)+(3*mxp)+3*(myp-2)+5*(mxp-2)*(myp-2))
cyCOUNT = int(mxp+4*(myp-2)+(3*mxp)+3*(myp-2)+5*(mxp-2)*(myp-2))
czCOUNT = int(mxp+(myp-2)+(3*mxp)+3*(myp-2)+5*(mxp-2)*(myp-2))

cxDATA  = np.zeros(( cxCOUNT ), dtype=np.double)
cyDATA  = np.zeros(( cyCOUNT ), dtype=np.double)
czDATA  = np.zeros(( czCOUNT ), dtype=np.double)

cxIND   = np.zeros((2, cxCOUNT), dtype=np.intc)
cyIND   = np.zeros((2, cyCOUNT), dtype=np.intc)
czIND   = np.zeros((2, czCOUNT), dtype=np.intc)


s2COEF  = np.zeros((mxp,myp), dtype=np.double)          # Smoothing strength profile for viscCOEF2


# INPUT and NORMALIZED VALUES : -------------------------
nstep, nstep_end = 0, 60001
timeSIM     = 0.            # simulated time
shearMAX    = 0.

visc1LOG    = True 
visc2LOG    = True 
gravLOG     = True
resiLOG     = False
    
viscCOEF1   = 1./150.       # coef of viscosity for general domain smoothing
viscCOEF2   = 1./20.        # coef of viscosity for specific smoothing near the origin
resiCOEF    = 1./1000.      # coef of resistivity 
    
vzMAX       = 1.0e-2        # max z-direction shear speed ratioed to normalized v
tACC        = 50.           # time required for shear to accelerate to max speed vzMAX
    
tempINIT    = 1.000e+0      # temperature (currently assumes system is isothermic)
adbtCONS    = 1.000e+0      # adibatic constant
gasR        = 1.624e-3      # gas constant
sunR        = 2.320e+1      # solar radius
surfGRAV    = 8.027e-4      # surface gravity 

cx, cy, cz, cs = 0., 0., 0., 0.

unifrmLOG = False            # Uniform grid: TRUE or FALSE

xmin, xmax = 0., 5.0
ymin, ymax = 0., 5.0
dx_min, dy_min = 0., 0. 

    
### ------------- MAIN FUNCTION ------------- ###
def main():
    global nstep, timeSIM, shearMAX

    set_grid()
    set_coef()
    initial()
    
    cfl             = 0.400
    velMAX          = calc_velMAX('1',cfl)
    dt              = cfl*min(dx_min,dy_min)/velMAX
    alpha           = 0.52          # semi-implicit coef (must staisfy 0.5<alpha<1.0)
    
    print("dx_min: {a}, dy_min: {b}, dt: {c}" .format(a=dx_min,b=dy_min,c=dt))
    
    
    # Init plotting
    record()
    #diagnostic_plotter(cfl)
    
    jzFILE      = open("jzfile.txt", 'w')
    shearFILE   = open("shearfile.txt", 'w')
    shearREC    = open("shearREC.txt", 'w')
    
    # Main Loop -------------------
    for n in range(nstep_end+1):
    
        time_start = time.time()
        stepon(dt, alpha, timeSIM, cfl)
        calc_velMAX('2',cfl)
        print("%.5f secs for step %i" %(time.time() - time_start,nstep))
                
        nstep       += 1
        timeSIM     += dt
        if (timeSIM < tACC):
            shearMAX += (timeSIM/tACC)*vzMAX*dt
        else:
            shearMAX += vzMAX*dt
            
        ### Diagnostics ###
        jzFILE.write("%s\n" %np.abs(np.min(cur[:,:,2])))
        shearFILE.write("%s\n" %shearMAX)
        
        if (n%300==0):
            print("Sim time: %.5f; Sim shear: %.5f" %(timeSIM,shearMAX))
            record()
            shearREC.write("%i, %s\n" %(nstep,shearMAX))
            #diagnostic_plotter(cfl)

    # End Main Loop ---------------
    
    jzFILE.close()
    shearFILE.close()
    shearREC.close()
    

### ---------------------------------------- ###

#
def set_grid():
    global dx_min, dy_min
    if unifrmLOG==True:
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
        dx_min = 1./20.
        dx_max = 15.80*dx_min
        dy_min = 1./11.
        dy_max = 12.0*dy_min

        
        print(dx_min, dx_max)
        
        xSPREAD, ySPREAD = 20., 26.
        
        # x:
        def dx_calc(i):
            return 0.5*(dx_max+dx_min) + 0.5*(dx_max-dx_min)*np.tanh((i-int(mxp/2))/xSPREAD)
            
        for i in range(mxp):
            dx[i] = dx_calc(i)
            xx[i] = xmin + sum(dx[0:i])
            
        for i in range(mxp+1):
            if (i==myp):
                xxx[i] = xxx[i-1]+dx[i-1]
            else:
                xxx[i] = xmin + sum(dx[0:i])
            
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
    
    xxfile = open("xxREC","wb")
    np.save(xxfile,xx)
    xxfile.close()
    
    yyfile = open("yyREC","wb")
    np.save(yyfile,yy)
    yyfile.close()
    

    #plt.plot(lz,dz)
    #plt.xlim(0,mzp)
    #plt.ylim(dz_min,dz_max*(1.02))
    #plt.show()

    #plt.plot(lx,xx)
    plt.plot(lx,dx)
    plt.xlim(0,mxp)
    #plt.ylim(xmin,xmax*(1.02))
    plt.show()
            
    
#
def set_coef():
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
        
    ### ------- BUILD C.S.C INDICES  ------ ###
    ### ----------------------------------- ###
    for s in range(1,4):
        print(s)
        if (s==1):
            cIND   = cxIND
        elif (s==2):
            cIND   = cyIND
        elif (s==3):
            cIND   = czIND
    
        k = 0
        for r in range(mxp*myp):
            ii = int(r/myp)
            jj = r - (ii*myp)
            # (a) Bottom B.C.: coef for v(i,0)
            if (r%myp==0):
                cIND[0,k], cIND[1,k]    = r, r # coef for v(i,0)
                k += 1
            # (b) Left-Side B.C.: coef for v(0,j)
            elif (r<myp-1):
                if (s==1 or s==3):                 
                    cIND[0,k], cIND[1,k]    = r, r                  # coef for v(0,j)
                    k += 1
                else:
                    cIND[0,k], cIND[1,k]    = r, r                  # coef for v(0,j)
                    k += 1
                    cIND[0,k], cIND[1,k]    = r, r+1                # coef for v(0,j+1)
                    k += 1
                    cIND[0,k], cIND[1,k]    = r, r-1                # coef for v(0,j-1)
                    k += 1
                    cIND[0,k], cIND[1,k]    = r, r+myp              # coef for v(0+1,j)
                    k += 1
            # (c) Top B.C.:coef for v(Max,j)            
            elif ((r+1)%myp==0):
                cIND[0,k], cIND[1,k]    = r, r              # coef for v(i,myp-1)
                k += 1
                cIND[0,k], cIND[1,k]    = r, r-1            # coef for v(i,myp-2)
                k += 1
                cIND[0,k], cIND[1,k]    = r, r-2            # coef for v(i,myp-3)
                k += 1
            # (d) Right-Side B.C.: coef for v(i,Max)
            elif (r>(mxp*myp-myp)):
                cIND[0,k], cIND[1,k]    = r, r              # coef for v(mxp-1,j)
                k += 1
                cIND[0,k], cIND[1,k]    = r, r-(1*myp)      # coef for v(mxp-2,j)
                k += 1
                cIND[0,k], cIND[1,k]    = r, r-(2*myp)      # coef for v(mxp-3,j)
                k += 1
            # (e) All Inner Points:
            else:
                cIND[0,k], cIND[1,k]    = r, r              # coef for v(i,j) 
                k += 1
                cIND[0,k], cIND[1,k]    = r, r+1            # coef for v(i,j+1) 
                k += 1
                cIND[0,k], cIND[1,k]    = r, r-1            # coef for v(i,j-1) 
                k += 1
                cIND[0,k], cIND[1,k]    = r, r+myp          # coef for v(i+1,j)
                k += 1
                cIND[0,k], cIND[1,k]    = r, r-myp          # coef for v(i-1,j)
                k += 1
    
    ### ------- SMOOTHING PROFILE 2 ------- ###
    ### ----------------------------------- ###
    for i in range(0,mxp):
        for j in range(0,myp):
            s2COEF[i,j] = 1.0 - 1.0*np.tanh(yy[j]/0.7)
#            s2COEF[i,j] = 1.0 - 1.0*np.tanh(np.sqrt(xx[i]**2 + yy[j]**2)/1.0)
            
    plt.pcolormesh(xxx,yyy,s2COEF.T)
    cb = plt.colorbar()
    plt.show()


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

            if (cA_help > cA_MAX): 
                cA_MAX = cA_help
                cA_xIND = i
                cA_yIND = j
            
    cLIGHT = 93.75
    if (np.abs(cA_MAX)>cLIGHT):
        diagnostic_plotter(cfl)
        print("Sim time: %.5f; Sim shear: %.5f" %(timeSIM,shearMAX))
        raise ValueError('A characteristic MHD speed exceeds speed of light.')
        
    print(" ",cA_MAX)
    print("  xIND: {a}, yIND: {b}" .format(a=cA_xIND,b=cA_yIND))

        
    if (num_IN=='1'):
        return cA_MAX
    else:
        pass
    
            
#
def stepon(dt, alpha, timeSIM, cfl):
    # The governing equations are solved by means of a semi-implicit method, which is a 
    # predictor-corrector type scheme (see [Harned $ Schnack (1986)]). The current the 
    # predictor-corrector iterates only once giving first-order accuracy in time. 
    
    # xOLD holds the current time step variable values at t_n.
    # xPRDCT holds the first-order prediction values of the variables advanced time 
    # step. 
    # x will now hold the updated values at t_(n+1)
    
    xOLD        = x.copy()
    xPRDCT      = np.zeros((mxp,myp,7), dtype=np.double)
    bxPRDCT     = np.zeros((mxp,myp),   dtype=np.double)
    byPRDCT     = np.zeros((mxp,myp),   dtype=np.double)
    curPRDCT    = np.zeros((mxp,myp,3), dtype=np.double)
    velRIGHT    = np.zeros((mxp,myp,3), dtype=np.double)    # right-side vel values in corrector step
    velHALF     = np.zeros((mxp,myp,3), dtype=np.double)    # the half-step velocity
    
    # (1): Predictor Step
    calc_innPRDCT(dt, alpha, xOLD, xPRDCT)              # inner points
    calc_outPRDCT(dt, alpha, xOLD, xPRDCT, timeSIM)     # outer/boundary points
    
    calc_bxby(xPRDCT[:,:,4], bxPRDCT, byPRDCT)
    calc_cur( xPRDCT[:,:,5], xPRDCT[:,:,4], curPRDCT)


    # (2): Calculate Max C Speeds Step Every Time Steps:
    if (nstep%1 == 0):
        cx, cy, cz, cs  = calc_cMAX(cfl, xPRDCT[:,:,0], xPRDCT[:,:,5], xPRDCT[:,:,6], bxPRDCT, byPRDCT)
        
        for s in range(1,4):
            calc_cMATRIX(s, dt)
    
    # (3): Corrector Step for Velocities
    for s in range(1,4):
        calc_correctRIGHT(s, velRIGHT[:,:,s-1], \
                             xOLD[:,:,s],       \
                             xPRDCT,            \
                             bxPRDCT,           \
                             byPRDCT,           \
                             curPRDCT,          \
                             dt )

        calc_correctLEFT(s, velRIGHT[:,:,s-1], dt)
        
        for i in range(0,mxp):
            for j in range(0,myp):
                velHALF[i,j,s-1] = (x[i,j,s] + xOLD[i,j,s])/2.

    # (4): Complete Time Advance for All Variables
    for s in range(7):
        if (s==0 or s==4 or s==5 or s==6):
            calc_innADVCD(s,velHALF,xOLD,xPRDCT,bxPRDCT,byPRDCT,dt)
            calc_outADVCD(s,velHALF,xOLD,xPRDCT,bxPRDCT,byPRDCT,dt) 
        else:
            pass     
            
    # (5): Update the final advanced/updated values of the Bx,By,J:
    calc_bxby(x[:,:,4], bx, by)
    calc_cur( x[:,:,5], x[:,:,4], cur)
    

#
def calc_innPRDCT(dt, alpha, xOLD, xPRDCT):
    # Calculates the first-order predictor values for all inner points.
    
    for i in range(1,mxp-1):
        for j in range(1,myp-1):
                
            # rhoPRDCT:
            fxp1 = xOLD[i+1,j,0]*xOLD[i+1,j,1]
            fx0  = xOLD[i+0,j,0]*xOLD[i+0,j,1]
            fxm1 = xOLD[i-1,j,0]*xOLD[i-1,j,1]
            
            fyp1 = xOLD[i,j+1,0]*xOLD[i,j+1,2]
            fy0  = xOLD[i,j+0,0]*xOLD[i,j+0,2]
            fym1 = xOLD[i,j-1,0]*xOLD[i,j-1,2]
            
            xPRDCT[i,j,0] = xOLD[i,j,0] - alpha*dt*( dfdxc('1',i,fxp1,fx0,fxm1)  \
                                                   + dfdyc('1',j,fyp1,fy0,fym1)  )
                                  
            # vxPRDCT
            fxp1 = xOLD[i+1,j,1]
            fx0  = xOLD[i+0,j,1]
            fxm1 = xOLD[i-1,j,1]
            
            fyp1 = xOLD[i,j+1,1]
            fy0  = xOLD[i,j+0,1]
            fym1 = xOLD[i,j-1,1]
            
            xPRDCT[i,j,1] = xOLD[i,j,1] - alpha*dt*(                                                 \
                                xOLD[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)                               \
                              + xOLD[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                               \
                              + (dfdxc('1',i,xOLD[i+1,j,6],xOLD[i+0,j,6],xOLD[i-1,j,6])/xOLD[i,j,0]) \
                              + ((cur[i,j,2]*by[i,j] - cur[i,j,1]*xOLD[i,j,5])/xOLD[i,j,0])          )
                              
            if (i==6 and j==2):
                rec1 = - alpha*dt*(xOLD[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1))
                rec2 = - alpha*dt*(xOLD[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1))
                rec3 = - alpha*dt*(dfdxc('1',i,xOLD[i+1,j,6],xOLD[i+0,j,6],xOLD[i-1,j,6])/xOLD[i,j,0])
                rec4 = - alpha*dt*((cur[i,j,2]*by[i,j] - cur[i,j,1]*xOLD[i,j,5])/xOLD[i,j,0]) 
    
            # vyPRDCT:
            fxp1 = xOLD[i+1,j,2]
            fx0  = xOLD[i+0,j,2]
            fxm1 = xOLD[i-1,j,2]
            
            fyp1 = xOLD[i,j+1,2]
            fy0  = xOLD[i,j+0,2]
            fym1 = xOLD[i,j-1,2]
            
            xPRDCT[i,j,2] = xOLD[i,j,2] - alpha*dt*(                                                    \
                                xOLD[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)                                  \
                              + xOLD[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                                  \
                              + ((cur[i,j,0]*xOLD[i,j,5] - cur[i,j,2]*bx[i,j])/xOLD[i,j,0])             )
                              
            if (gravLOG == True):
                delRHO_p1 = xOLD[i,j+1,0] - rhoINIT[i,j+1]
                delRHO_00 = xOLD[i,j+0,0] - rhoINIT[i,j+0]
                delRHO_m1 = xOLD[i,j-1,0] - rhoINIT[i,j-1]
                
                xPRDCT[i,j,2] -= alpha*dt*( gasR*tempINIT*dfdyc('1',j,delRHO_p1,delRHO_00,delRHO_m1)    \
                                          + (delRHO_00*surfGRAV*(sunR**2)/(sunR + yy[j])**2)          ) \
                                          /xOLD[i,j,0]
            else:
                xPRDCT[i,j,2] -= alpha*dt*dfdyc('1',j,xOLD[i,j+1,6],xOLD[i,j+0,6],xOLD[i,j-1,6])/xOLD[i,j,0]
                
            # vzPRDCT:
            fxp1 = xOLD[i+1,j,3]
            fx0  = xOLD[i+0,j,3]
            fxm1 = xOLD[i-1,j,3]
            
            fyp1 = xOLD[i,j+1,3]
            fy0  = xOLD[i,j+0,3]
            fym1 = xOLD[i,j-1,3]
            
            xPRDCT[i,j,3] = xOLD[i,j,3] - alpha*dt*(                                                \
                                          xOLD[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)                    \
                                        + xOLD[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                    \
                                        + ((cur[i,j,1]*bx[i,j] - cur[i,j,0]*by[i,j])/xOLD[i,j,0])   )

            # psiPRDCT:
            fxp1 = xOLD[i+1,j,4]
            fx0  = xOLD[i+0,j,4]
            fxm1 = xOLD[i-1,j,4]
            
            fyp1 = xOLD[i,j+1,4]
            fy0  = xOLD[i,j+0,4]
            fym1 = xOLD[i,j-1,4]
            
            xPRDCT[i,j,4] = xOLD[i,j,4] - alpha*dt*( xOLD[i,j,1]*dfdxc('1',i,fxp1,fx0,fxm1)  \
                                                   + xOLD[i,j,2]*dfdyc('1',j,fyp1,fy0,fym1)  )
                              
            # BzPRDCT:
            fxp1 = xOLD[i+1,j,3]*bx[i+1,j] - xOLD[i+1,j,1]*xOLD[i+1,j,5]
            fx0  = xOLD[i+0,j,3]*bx[i+0,j] - xOLD[i+0,j,1]*xOLD[i+0,j,5]
            fxm1 = xOLD[i-1,j,3]*bx[i-1,j] - xOLD[i-1,j,1]*xOLD[i-1,j,5]
            
            fyp1 = xOLD[i,j+1,3]*by[i,j+1] - xOLD[i,j+1,2]*xOLD[i,j+1,5]
            fy0  = xOLD[i,j+0,3]*by[i,j+0] - xOLD[i,j+0,2]*xOLD[i,j+0,5]
            fym1 = xOLD[i,j-1,3]*by[i,j-1] - xOLD[i,j-1,2]*xOLD[i,j-1,5]
            
            xPRDCT[i,j,5] = xOLD[i,j,5] + alpha*dt*( dfdxc('1',i,fxp1,fx0,fxm1)  \
                                                   + dfdyc('1',j,fyp1,fy0,fym1)  )
                              
            # pressurePRDCT:
            xPRDCT[i,j,6] = xPRDCT[i,j,0]*gasR*tempINIT     


#
def calc_outPRDCT(dt, alpha, xOLD, xPRDCT, timeSIM):
    # Calculates the first-order predictor values for all boundary points.
    # Order: Left-side (symmetric boundary); Top (Open); Right-side (Open); Bottom (Fixed/Open)
    
    
    # (1): Left-side boundary ----------------------------
    for j in range(1,myp-1):
    
        # rhoPRDCT:
        fxp1 = +xOLD[1,j,1]
        fx0  = +xOLD[0,j,1]
        fxm1 = -xOLD[1,j,1]

        fyp1 = xOLD[0,j+1,0]*xOLD[0,j+1,2]
        fy0  = xOLD[0,j+0,0]*xOLD[0,j+0,2]
        fym1 = xOLD[0,j-1,0]*xOLD[0,j-1,2]
            
        xPRDCT[0,j,0] = xOLD[0,j,0] - alpha*dt*( (dfdxc('1',0,fxp1,fx0,fxm1)*xOLD[0,j,0])    \
                                                + dfdyc('1',j,fyp1,fy0,fym1)                 )
        
        # vxPRDCT
        xPRDCT[0,j,1] = +0.0
        
        # vyPRDCT:
        fyp1 = xOLD[0,j+1,2]
        fy0  = xOLD[0,j+0,2]
        fym1 = xOLD[0,j-1,2]
            
        xPRDCT[0,j,2] = xOLD[0,j,2] - alpha*dt*(                                                    \
                                xOLD[0,j,2]*dfdyc('1',j,fyp1,fy0,fym1)                              \
                             + ((cur[0,j,0]*xOLD[0,j,5] - cur[0,j,2]*bx[0,j])/xOLD[0,j,0])          )
                              
        if (gravLOG == True):
            delRHO_p1 = xOLD[0,j+1,0] - rhoINIT[0,j+1]
            delRHO_00 = xOLD[0,j+0,0] - rhoINIT[0,j+0]
            delRHO_m1 = xOLD[0,j-1,0] - rhoINIT[0,j-1]
                
            xPRDCT[0,j,2] -= alpha*dt*( gasR*tempINIT*dfdyc('1',j,delRHO_p1,delRHO_00,delRHO_m1)    \
                                      + (delRHO_00*surfGRAV*(sunR**2)/(sunR + yy[j])**2)          ) \
                                        /xOLD[0,j,0]
        else:
            xPRDCT[0,j,2] -= alpha*dt*dfdyc('1',j,xOLD[0,j+1,6],xOLD[0,j+0,6],xOLD[0,j-1,6])/xOLD[0,j,0]
        
        # vzPRDCT:
        xPRDCT[0,j,3] = +0.0
        
        # psiPRDCT:
        fyp1 = xOLD[0,j+1,4]
        fy0  = xOLD[0,j+0,4]
        fym1 = xOLD[0,j-1,4]
            
        xPRDCT[0,j,4] = xOLD[0,j,4] - alpha*dt*( xOLD[0,j,2]*dfdyc('1',j,fyp1,fy0,fym1) )

        # BzPRDCT:
        dervx_1 = dfdxc('1',0,xOLD[1,j,3],0.0,-xOLD[1,j,3])
        dervx_2 = dfdxc('1',0,xOLD[1,j,1],0.0,-xOLD[1,j,1])

        dervy_1 = dfdyc('1',j,xOLD[0,j+1,5],xOLD[0,j+0,5],xOLD[0,j-1,5])
        dervy_2 = dfdyc('1',j,xOLD[0,j+1,2],xOLD[0,j+0,2],xOLD[0,j-1,2])
            
        xPRDCT[0,j,5] = xOLD[0,j,5] + alpha*dt*(    dervx_1*bx[0,j]         \
                                                -   dervx_2*xOLD[0,j,5]     \
                                                -   dervy_1*xOLD[0,j,2]     \
                                                -   dervy_2*xOLD[0,j,5]     )

        # pressurePRDCT:
        xPRDCT[0,j,6] = xPRDCT[0,j,0]*gasR*tempINIT
        
    # (2): Top boundary ----------------------------
    for i in range(0,mxp-1):
    
        # rhoPRDCT:
        fym1 = xPRDCT[i,myp-2,0] - xOLD[i,myp-2,0]
        fym2 = xPRDCT[i,myp-3,0] - xOLD[i,myp-3,0]
        
        xPRDCT[i,myp-1,0] = xOLD[i,myp-1,0] - ((dymCOEF[myp-1,0,1]*fym1     \
                                              + dymCOEF[myp-1,0,2]*fym2)    \
                                               /dymCOEF[myp-1,0,0] )
        
        # vxPRDCT
        xPRDCT[i,myp-1,1] = - ( (dymCOEF[myp-1,0,1]*xPRDCT[i,myp-2,1]   \
                               + dymCOEF[myp-1,0,2]*xPRDCT[i,myp-3,1])  \
                                /dymCOEF[myp-1,0,0] )
        
        # vyPRDCT:
        xPRDCT[i,myp-1,2] = - ( (dymCOEF[myp-1,0,1]*xPRDCT[i,myp-2,2]   \
                               + dymCOEF[myp-1,0,2]*xPRDCT[i,myp-3,2])  \
                                /dymCOEF[myp-1,0,0] )
        
        # vzPRDCT:
        xPRDCT[i,myp-1,3] = - ( (dymCOEF[myp-1,0,1]*xPRDCT[i,myp-2,3]   \
                               + dymCOEF[myp-1,0,2]*xPRDCT[i,myp-3,3])  \
                                /dymCOEF[myp-1,0,0] )
        
        # psiPRDCT:
        if (i==0):
            derv_term  = 0.0
            
            if (xOLD[i,myp-1,2] > 0.):
                fy0  = xOLD[0,myp-1,4]
                fym1 = xOLD[0,myp-2,4]
                fym2 = xOLD[0,myp-3,4]
                
                derv_term += xOLD[i,myp-1,2] * dfdym('1',myp-1, fy0, fym1, fym2 )
                
            xPRDCT[i,myp-1,4] = xOLD[i,myp-1,4] - alpha*dt*derv_term
        
        else:
            fxp1 = xOLD[i+1,myp-1,4]
            fx0  = xOLD[i+0,myp-1,4]
            fxm1 = xOLD[i-1,myp-1,4]
            
            derv_term = xOLD[i,myp-1,1] * dfdxc('1',i, fxp1, fx0, fxm1 )
            
            if (xOLD[i,myp-1,2] > 0.):
                fy0  = xOLD[i,myp-1,4]
                fym1 = xOLD[i,myp-2,4]
                fym2 = xOLD[i,myp-3,4]
                
                derv_term += xOLD[i,myp-1,2] * dfdym('1',myp-1, fy0, fym1, fym2 )
                
            xPRDCT[i,myp-1,4] = xOLD[i,myp-1,4] - alpha*dt*derv_term
            
        # BzPRDCT:
        xPRDCT[i,myp-1,5] = - ( (dymCOEF[myp-1,0,1]*xPRDCT[i,myp-2,5]   \
                               + dymCOEF[myp-1,0,2]*xPRDCT[i,myp-3,5])  \
                                /dymCOEF[myp-1,0,0] )

        # pressurePRDCT:
        xPRDCT[i,myp-1,6] = xPRDCT[i,myp-1,0]*gasR*tempINIT
        
        
    # (3): Right-side boundary ----------------------------
    for j in range(1,myp):
    
        # rhoPRDCT:
        fxm1 = xPRDCT[mxp-2,j,0] - xOLD[mxp-2,j,0]
        fxm2 = xPRDCT[mxp-3,j,0] - xOLD[mxp-3,j,0]
        
        xPRDCT[mxp-1,j,0] = xOLD[mxp-1,j,0] - ((dxmCOEF[mxp-1,0,1]*fxm1     \
                                              + dxmCOEF[mxp-1,0,2]*fxm2)    \
                                               /dxmCOEF[mxp-1,0,0] )
        
        # vxPRDCT
        xPRDCT[mxp-1,j,1] = - ( (dxmCOEF[mxp-1,0,1]*xPRDCT[mxp-2,j,1]   \
                               + dxmCOEF[mxp-1,0,2]*xPRDCT[mxp-3,j,1])  \
                                /dxmCOEF[mxp-1,0,0] )
        
        # vyPRDCT:
        xPRDCT[mxp-1,j,2] = - ( (dxmCOEF[mxp-1,0,1]*xPRDCT[mxp-2,j,2]   \
                               + dxmCOEF[mxp-1,0,2]*xPRDCT[mxp-3,j,2])  \
                                /dxmCOEF[mxp-1,0,0] )
        
        # vzPRDCT:
        xPRDCT[mxp-1,j,3] = - ( (dxmCOEF[mxp-1,0,1]*xPRDCT[mxp-2,j,3]   \
                               + dxmCOEF[mxp-1,0,2]*xPRDCT[mxp-3,j,3])  \
                                /dxmCOEF[mxp-1,0,0] )
        
        # psiPRDCT:
        if (j==0):
            # j==0 value will be calculated in bottom boundary.
            pass
        elif (j==myp-1):
            derv_term = 0.0
            
            if (xOLD[mxp-1,j,2] > 0.):
                fy0  = xOLD[mxp-1,j+0,4]
                fym1 = xOLD[mxp-1,j-1,4]
                fym2 = xOLD[mxp-1,j-2,4]
            
                derv_term += xOLD[mxp-1,j,2] * dfdym('1',j, fy0, fym1, fym2 )
            
            if (xOLD[mxp-1,j,1] > 0.):
                fx0  = xOLD[mxp-1,j,4]
                fxm1 = xOLD[mxp-2,j,4]
                fxm2 = xOLD[mxp-3,j,4]
                
                derv_term += xOLD[mxp-1,j,1] * dfdxm('1',mxp-1, fx0, fxm1, fxm2 )
                
            xPRDCT[mxp-1,j,4] = xOLD[mxp-1,j,4] - alpha*dt*derv_term       
        else:
            fyp1 = xOLD[mxp-1,j+1,4]
            fy0  = xOLD[mxp-1,j+0,4]
            fym1 = xOLD[mxp-1,j-1,4]
            
            derv_term  = xOLD[mxp-1,j,2] * dfdyc('1',j, fyp1, fy0, fym1 )
            
            if (xOLD[mxp-1,j,1] > 0.):
                fx0  = xOLD[mxp-1,j,4]
                fxm1 = xOLD[mxp-2,j,4]
                fxm2 = xOLD[mxp-3,j,4]
                
                derv_term += xOLD[mxp-1,j,1] * dfdxm('1',mxp-1, fx0, fxm1, fxm2 )
                
            xPRDCT[mxp-1,j,4] = xOLD[mxp-1,j,4] - alpha*dt*derv_term

        # BzPRDCT:
        xPRDCT[mxp-1,j,5] = - ( (dxmCOEF[mxp-1,0,1]*xPRDCT[mxp-2,j,5]       \
                               + dxmCOEF[mxp-1,0,2]*xPRDCT[mxp-3,j,5])      \
                                /dxmCOEF[mxp-1,0,0] )

        # pressurePRDCT:
        xPRDCT[mxp-1,j,6] = xPRDCT[mxp-1,j,0]*gasR*tempINIT
        
        
    # (4): Bottom boundary ----------------------------
    for i in range(0,mxp):
        
        # rhoPRDCT:
        xPRDCT[i,0,0] = xOLD[i,0,0] - alpha*dt*(xOLD[i,0,0]*dfdyp('1',0,xOLD[i,2,2],xOLD[i,1,2],xOLD[i,0,2]))
        
        # vxPRDCT
        xPRDCT[i,0,1] = +0.0
        
        # vyPRDCT:
        xPRDCT[i,0,2] = +0.0
        
        # vzPRDCT:
        if (timeSIM <= tACC):
            velSHEAR = vzMAX*xx[i]*np.exp(0.5*(1.0 - xx[i]**2))
            xPRDCT[i,0,3] = xOLD[i,0,3] + alpha*dt*(velSHEAR/tACC)
        else:
            xPRDCT[i,0,3] = xOLD[i,0,3]
        
        # psiPRDCT:
        xPRDCT[i,0,4] = xOLD[i,0,4]

        # BzPRDCT:
        if (i==0):
            derv_xa = dfdxc('1',0,xOLD[1,0,3],xOLD[0,0,3],-xOLD[1,0,3])
            derv_ya = dfdyp('1',0,xOLD[0,2,2],xOLD[0,1,2],+xOLD[0,0,2])
            
            xPRDCT[i,0,5] = xOLD[i,0,5] + alpha*dt*(  derv_xa*bx[0,0]       \
                                                   -  derv_ya*xOLD[0,0,5]   )

        elif (i==mxp-1):
            derv_xa = dfdxm('1',i,bx[i+0,0],bx[i-1,0],bx[i-2,0])
            derv_xb = dfdxm('1',i,xOLD[i+0,0,3],xOLD[i-1,0,3],xOLD[i-2,0,3])
            
            derv_ya = dfdyp('1',0,by[i,2],by[i,1],by[i,0])
            derv_yb = dfdyp('1',0,xOLD[i,2,3],xOLD[i,1,3],xOLD[i,0,3])
            derv_yc = dfdyp('1',0,xOLD[i,2,2],xOLD[i,1,2],xOLD[i,0,2])
            
            xPRDCT[i,0,5] = xOLD[i,0,5] + alpha*dt*(    derv_xa*xOLD[i,0,3]     \
                                                    +   derv_xb*bx[i,0]         \
                                                    +   derv_ya*xOLD[i,0,3]     \
                                                    +   derv_yb*by[i,0]         \
                                                    -   derv_yc*xOLD[i,0,5]     )
                                               
        else:
            derv_xa = dfdxc('1',i,bx[i+1,0],bx[i+0,0],bx[i-1,0])
            derv_xb = dfdxc('1',i,xOLD[i+1,0,3],xOLD[i+0,0,3],xOLD[i-1,0,3])
            
            derv_ya = dfdyp('1',0,by[i,2],by[i,1],by[i,0])
            derv_yb = dfdyp('1',0,xOLD[i,2,3],xOLD[i,1,3],xOLD[i,0,3])
            derv_yc = dfdyp('1',0,xOLD[i,2,2],xOLD[i,1,2],xOLD[i,0,2])
            
            xPRDCT[i,0,5] = xOLD[i,0,5] + alpha*dt*(    derv_xa*xOLD[i,0,3]     \
                                                    +   derv_xb*bx[i,0]         \
                                                    +   derv_ya*xOLD[i,0,3]     \
                                                    +   derv_yb*by[i,0]         \
                                                    -   derv_yc*xOLD[i,0,5]     )
        
        # pressurePRDCT:
        xPRDCT[i,0,6] = xPRDCT[i,0,0]*gasR*tempINIT
        
        
#
def calc_bxby(psi_IN, bx_IN, by_IN):
    # Calculates the first-order predicted values for Bx and By. 
    
    w = np.zeros((mxp,myp), dtype=np.double)
    
    for i in range(mxp):
        for j in range(myp):
            w[i,j] = psi_IN[i,j] - psiINIT[i,j]
            
    # Calculate dBx:
    for i in range(mxp):
        for j in range(myp):
            if (j==0):
                bx_IN[i,j] = - dfdyp('1',j,w[i,j+2],w[i,j+1],w[i,j+0])
            elif (j==myp-1):
                bx_IN[i,j] = - dfdym('1',j,w[i,j+0],w[i,j-1],w[i,j-2])
            else:
                bx_IN[i,j] = - dfdyc('1',j,w[i,j+1],w[i,j+0],w[i,j-1])
                
    # Calculate dBy:
    for i in range(mxp):
        for j in range(myp):
            if (j==0):
                by_IN[i,j] = 0.0
            else:
                if (i==0):
                    by_IN[i,j] = + 0.0
                elif (i==mxp-1):
                    by_IN[i,j] = + dfdxm('1',i,w[i+0,j],w[i-1,j],w[i-2,j])
                else:
                    by_IN[i,j] = + dfdxc('1',i,w[i+1,j],w[i+0,j],w[i-1,j])
                    
    # Add the initial Bx and By values to dBx and dBy:
    for i in range(mxp):
        for j in range(myp):
            bx_IN[i,j] += bINIT[i,j,0]
            by_IN[i,j] += bINIT[i,j,1]
            

#    
def calc_cur(bz_IN, psi_IN, cur_IN):
    # Calculates the current values of the current density components.

    w = np.zeros((mxp,myp), dtype=np.double)
    
    for i in range(mxp):
        for j in range(myp):
            w[i,j] = psi_IN[i,j] - psiINIT[i,j]
    
    # Calculate jx: ---------------------------------
    for i in range(mxp):
        for j in range(myp):
            if (j==0):
                cur_IN[i,j,0] = + dfdyp('1',j,bz_IN[i,j+2],bz_IN[i,j+1],bz_IN[i,j+0])
            elif (j==myp-1):
                cur_IN[i,j,0] = + dfdym('1',j,bz_IN[i,j+0],bz_IN[i,j-1],bz_IN[i,j-2])
            else:
                cur_IN[i,j,0] = + dfdyc('1',j,bz_IN[i,j+1],bz_IN[i,j+0],bz_IN[i,j-1])
    
    # Calculate jy: ---------------------------------
    for i in range(mxp):
        for j in range(myp):
            if (i==0):
                cur_IN[i,j,1] = + 0.0
            elif (i==mxp-1):
                cur_IN[i,j,1] = - dfdxm('1',i,bz_IN[i+0,j],bz_IN[i-1,j],bz_IN[i-2,j])
            else:
                cur_IN[i,j,1] = - dfdxc('1',i,bz_IN[i+1,j],bz_IN[i+0,j],bz_IN[i-1,j])
                            
    # Calculate jz: ---------------------------------
    for i in range(1,mxp-1):
        for j in range(1,myp-1):
            cur_IN[i,j,2] =   dfdxc('2',i,w[i+1,j],w[i+0,j],w[i-1,j]) \
                            + dfdyc('2',j,w[i,j+1],w[i,j+0],w[i,j-1]) 
                                  
        # jz left-side boundary:
    i = 0
    for j in range(1,myp):
        if (j==myp-1):
            cur_IN[i,j,2] =   dfdxc('2',i,w[i+1,j],w[i+0,j],w[i+1,j]) \
                            + dfdym('2',j,w[i,j+0],w[i,j-1],w[i,j-2])
        else:
            cur_IN[i,j,2] =   dfdxc('2',i,w[i+1,j],w[i+0,j],w[i+1,j]) \
                            + dfdyc('2',j,w[i,j+1],w[i,j+0],w[i,j-1])
        
        # jz top boundary:
    j = myp-1
    for i in range(1,mxp):
        if (i==mxp-1):
            cur_IN[i,j,2] =   dfdxm('2',i,w[i+0,j],w[i-1,j],w[i-2,j]) \
                            + dfdym('2',j,w[i,j+0],w[i,j-1],w[i,j-2])
        else:
            cur_IN[i,j,2] =   dfdxc('2',i,w[i+1,j],w[i+0,j],w[i-1,j]) \
                            + dfdym('2',j,w[i,j+0],w[i,j-1],w[i,j-2])
                              
        # jz right-side boundary:
    i = mxp-1
    for j in range(1,myp-1):
        cur_IN[i,j,2] =   dfdxm('2',i,w[i+0,j],w[i-1,j],w[i-2,j]) \
                        + dfdyc('2',j,w[i,j+1],w[i,j+0],w[i,j-1]) 
                              
        # jz bottom boundary:
    j = 0
    for i in range(0,mxp):
        if (i==0):
            cur_IN[i,j,2] = + dfdyc('2',j,w[i,j+2],w[i,j+1],w[i,j+0])
                                  

#
#def calc_cMAX(rhoOLD, bzOLD, presOLD):
def calc_cMAX(cfl, rhoPRDCT, bzPRDCT, presPRDCT, bxPRDCT, byPRDCT):
    # Calculates the maximum charactersitic speed values: cx, cy, cz, cs
    
    cxMAX, cyMAX, czMAX, csMAX = 0., 0., 0., 0.
    
    for i in range(mxp):
        for j in range(myp):
            cx_help = np.abs(bxPRDCT[i,j]/np.sqrt(rhoPRDCT[i,j]))
            cy_help = np.abs(byPRDCT[i,j]/np.sqrt(rhoPRDCT[i,j]))
            cz_help = np.abs(bzPRDCT[i,j]/np.sqrt(rhoPRDCT[i,j]))
            cs_help = np.sqrt(adbtCONS*presPRDCT[i,j]/rhoPRDCT[i,j])
        
            if (cx_help > cxMAX): cxMAX = cx_help
            if (cy_help > cyMAX): cyMAX = cy_help
            if (cz_help > czMAX): czMAX = cz_help
            if (cs_help > csMAX): csMAX = cs_help
            
    print("   %.5f,%.5f,%.5f,%.5f" %(cxMAX, cyMAX, czMAX, csMAX))
    
    cLIGHT = 93.75
    if (np.abs(cxMAX)>cLIGHT or np.abs(cyMAX)>cLIGHT or    \
        np.abs(czMAX)>cLIGHT or np.abs(csMAX)>cLIGHT):
        diagnostic_plotter(cfl)
        print("Sim time: %.5f; Sim shear: %.5f" %(timeSIM,shearMAX))
        raise ValueError('A characteristic MHD speed exceeds speed of light.')
    
    return cxMAX, cyMAX, czMAX, csMAX
    

#
def calc_cMATRIX(s, dt):
    # Builds the 3 C.S.C coefficient matrices needed to solve the left-side of the velocity corrector step. 
    
    if (s==1):
        cCOEF1 = cy**2 + cz**2 + cs**2
        cCOEF2 = cy**2
        cDATA  = cxDATA
    elif (s==2):
        cCOEF1 = cx**2
        cCOEF2 = cz**2 + cx**2 + cs**2
        cDATA  = cyDATA
    elif (s==3):
        cCOEF1 = cx**2
        cCOEF2 = cy**2
        cDATA  = czDATA

    # Full coefficient functions for coefficient matrix for advanced v_ij (ex. v0p = v_(i,j+1))
    def v00_COEF(ii_in, jj_in, dt):
        visc1TERM, visc2TERM = 0., 0.
        semiTERM = 1.0 - (dt**2)*(cCOEF1*dxcCOEF[ii_in,1,1] + cCOEF2*dycCOEF[jj_in,1,1])
        if (visc1LOG==True):
            visc1TERM = - dt*viscCOEF1*(dxcCOEF[ii_in,1,1] + dycCOEF[jj_in,1,1])
        if (visc2LOG==True):
            visc2TERM = - dt*viscCOEF2*s2COEF[ii_in,jj_in]*(dxcCOEF[ii_in,1,1] + dycCOEF[jj_in,1,1])
        return semiTERM + visc1TERM + visc2TERM
        
    def vp0_COEF(ii_in, jj_in, dt):
        visc1TERM, visc2TERM = 0., 0.
        semiTERM = - (dt**2)*cCOEF1*dxcCOEF[ii_in,1,0]
        if (visc1LOG==True):
            visc1TERM = - dt*viscCOEF1*dxcCOEF[ii_in,1,0]
        if (visc2LOG==True):
            visc2TERM = - dt*viscCOEF2*s2COEF[ii_in,jj_in]*dxcCOEF[ii_in,1,0]
        return semiTERM + visc1TERM + visc2TERM
    
    def vm0_COEF(ii_in, jj_in, dt):
        visc1TERM, visc2TERM = 0., 0.
        semiTERM = - (dt**2)*cCOEF1*dxcCOEF[ii_in,1,2]
        if (visc1LOG==True):
            visc1TERM = - dt*viscCOEF1*dxcCOEF[ii_in,1,2]
        if (visc2LOG==True):
            visc2TERM = - dt*viscCOEF2*s2COEF[ii_in,jj_in]*dxcCOEF[ii_in,1,2]
        return semiTERM + visc1TERM + visc2TERM
    
    def v0p_COEF(ii_in, jj_in, dt):
        visc1TERM, visc2TERM = 0., 0.
        semiTERM = - (dt**2)*cCOEF2*dycCOEF[jj_in,1,0]
        if (visc1LOG==True):
            visc1TERM = - dt*viscCOEF1*dycCOEF[jj_in,1,0]
        if (visc2LOG==True):
            visc2TERM = - dt*viscCOEF2*s2COEF[ii_in,jj_in]*dycCOEF[jj_in,1,0]
        return semiTERM + visc1TERM + visc2TERM
    
    def v0m_COEF(ii_in, jj_in, dt):
        visc1TERM, visc2TERM = 0., 0.
        semiTERM = - (dt**2)*cCOEF2*dycCOEF[jj_in,1,2]
        if (visc1LOG==True):
            visc1TERM = - dt*viscCOEF1*dycCOEF[jj_in,1,2]
        if (visc2LOG==True):
            visc2TERM = - dt*viscCOEF2*s2COEF[ii_in,jj_in]*dycCOEF[jj_in,1,2]
        return semiTERM + visc1TERM + visc2TERM
        
    # Full coefficient functions for coefficient matrix for advanced v_ij (ex. v0p = v_(i,j+1)) at LS
    def vp0LS_COEF(ii_in, jj_in, dt):
        visc1TERM, visc2TERM = 0., 0.
        semiTERM = - (dt**2)*cCOEF1*(dxcCOEF[ii_in,1,0] + dxcCOEF[ii_in,1,2])
        if (visc1LOG==True):
            visc1TERM = - dt*viscCOEF1*(dxcCOEF[ii_in,1,0] + dxcCOEF[ii_in,1,2]) 
        if (visc2LOG==True):
            visc2TERM = - dt*viscCOEF2*s2COEF[ii_in,jj_in]*(dxcCOEF[ii_in,1,0] + dxcCOEF[ii_in,1,2]) 
        return semiTERM + visc1TERM + visc2TERM
                

    # (3) Build coefficient matrix:
    k = 0
    for r in range(mxp*myp):
        ii = int(r/myp)
        jj = r - (ii*myp)
        # (a) Bottom B.C.: coef for v(i,0)
        if (r%myp==0):
            cDATA[k]                    = 1.        # coef for v(i,0)
            k += 1
        # (b) Left-Side B.C.: coef for v(0,j)
        elif (r<myp-1):
            if (s==1 or s==3):                 
                cDATA[k]                = 1.        # coef for v(0,j)
                k += 1
            else:
                cDATA[k]                = v00_COEF(ii,jj,dt)    # coef for v(0,j)
                k += 1
                cDATA[k]                = v0p_COEF(ii,jj,dt)    # coef for v(0,j+1)
                k += 1
                cDATA[k]                = v0m_COEF(ii,jj,dt)    # coef for v(0,j-1)
                k += 1
                cDATA[k]                = vp0LS_COEF(ii,jj,dt)  # coef for v(0+1,j)
                k += 1
        # (c) Top B.C.:coef for v(Max,j)            
        elif ((r+1)%myp==0):
            cDATA[k]                = dymCOEF[jj,0,0]   # coef for v(i,myp-1)
            k += 1
            cDATA[k]                = dymCOEF[jj,0,1]   # coef for v(i,myp-2)
            k += 1
            cDATA[k]                = dymCOEF[jj,0,2]   # coef for v(i,myp-3)
            k += 1
        # (d) Right-Side B.C.: coef for v(i,Max)
        elif (r>(mxp*myp-myp)):
            cDATA[k]                = dxmCOEF[ii,0,0]   # coef for v(mxp-1,j)
            k += 1
            cDATA[k]                = dxmCOEF[ii,0,1]   # coef for v(mxp-2,j)
            k += 1
            cDATA[k]                = dxmCOEF[ii,0,2]   # coef for v(mxp-3,j)
            k += 1
        # (e) All Inner Points:
        else:
            cDATA[k]                = v00_COEF(ii,jj,dt)    # coef for v(i,j)
            k += 1
            cDATA[k]                = v0p_COEF(ii,jj,dt)    # coef for v(i,j+1)
            k += 1
            cDATA[k]                = v0m_COEF(ii,jj,dt)    # coef for v(i,j-1)
            k += 1
            cDATA[k]                = vp0_COEF(ii,jj,dt)    # coef for v(i+1,j)
            k += 1
            cDATA[k]                = vm0_COEF(ii,jj,dt)    # coef for v(i-1,j)
            k += 1


#
def calc_correctRIGHT(s, vRIGHT,    v_old,       \
                         xPRDCT,  bxPRDCT,       \
                         byPRDCT, curPRDCT,      \
                         dt ):
    # Calculates explicitly the right-side of the corrector equations for the velocities. 
    # s in the index referring to the velocities in the xOLD and xPRDCT arrays. 
    
    for i in range(1,mxp-1):
        for j in range(1,myp-1):
        
            # Load predictor derivatives values:
            derv1_v_x = dfdxc('1',i,xPRDCT[i+1,j,s],xPRDCT[i+0,j,s],xPRDCT[i-1,j,s])
            derv1_v_y = dfdyc('1',j,xPRDCT[i,j+1,s],xPRDCT[i,j+0,s],xPRDCT[i,j-1,s])
    
            # Load right-side corrector of old derivatives values:
            derv2_v_x = dfdxc('2',i,v_old[i+1,j],v_old[i+0,j],v_old[i-1,j])
            derv2_v_y = dfdyc('2',j,v_old[i,j+1],v_old[i,j+0],v_old[i,j-1])
            
            # Calculate right-side value:
            v_right_help  = v_old[i,j] - dt*(xPRDCT[i,j,1]*derv1_v_x + xPRDCT[i,j,2]*derv1_v_y)
            
            if (s==1):
                derv_p = dfdxc('1',i,xPRDCT[i+1,j,6],xPRDCT[i+0,j,6],xPRDCT[i-1,j,6])
                term_help = derv_p - (curPRDCT[i,j,1]*xPRDCT[i,j,5] - curPRDCT[i,j,2]*byPRDCT[i,j])
                c_coef = cy**2 + cz**2 + cs**2
            
                v_right_help -= dt*term_help/xPRDCT[i,j,0]
                v_right_help -= dt**2*(c_coef*derv2_v_x + (cy**2)*derv2_v_y)
                
                if (i==6 and j==2):
                    rec1 = - dt*(xPRDCT[i,j,1]*derv1_v_x)
                    rec2 = - dt*(xPRDCT[i,j,2]*derv1_v_y)
                    rec3 = - dt*(derv_p/xPRDCT[i,j,0])
                    rec4 = - dt*((curPRDCT[i,j,2]*byPRDCT[i,j] - curPRDCT[i,j,1]*xPRDCT[i,j,5])/xPRDCT[i,j,0]) 
                
            elif (s==2):
                term_help = - (curPRDCT[i,j,2]*bxPRDCT[i,j] - curPRDCT[i,j,0]*xPRDCT[i,j,5])
                c_coef = cz**2 + cx**2 + cs**2
            
                v_right_help -= dt*term_help/xPRDCT[i,j,0]
                v_right_help -= dt**2*((cx**2)*derv2_v_x + c_coef*derv2_v_y)
                
                if (gravLOG==True):
                    delRHO_p1 = xPRDCT[i,j+1,0] - rhoINIT[i,j+1]
                    delRHO_00 = xPRDCT[i,j+0,0] - rhoINIT[i,j+0]
                    delRHO_m1 = xPRDCT[i,j-1,0] - rhoINIT[i,j-1]
                                        
                    v_right_help -= dt*( gasR*tempINIT*dfdyc('1',j,delRHO_p1,delRHO_00,delRHO_m1)   \
                                       + (delRHO_00*surfGRAV*(sunR**2)/(sunR + yy[j])**2)         ) \
                                        /xPRDCT[i,j,0]
                else:
                    derv_p = dfdyc('1',j,xPRDCT[i,j+1,6],xPRDCT[i,j+0,6],xPRDCT[i,j-1,6])
                    v_right_help -= dt*derv_p/xPRDCT[i,j,0]
                    
            elif (s==3):
                term_help = - (curPRDCT[i,j,0]*byPRDCT[i,j] - curPRDCT[i,j,1]*bxPRDCT[i,j])
                
                v_right_help -= dt*term_help/xPRDCT[i,j,0]
                v_right_help -= dt**2*((cx**2)*derv2_v_x + (cy**2)*derv2_v_y)
                
            else:
                print("Superfluous iteration in calc_correctRIGHT: {a}" .format(a=s))
                
            # Load calculated right-side value into vRIGHT array:
            vRIGHT[i,j] = v_right_help
    
    # (2) Calculate the left-side boundary for the right side of the corrector step for vy.
    if (s==2):
        for j in range(1,myp-1):
            # Load predictor and corrector of old y derivatives values:
            derv1_v_y = dfdyc('1',j,xPRDCT[0,j+1,s],xPRDCT[0,j+0,s],xPRDCT[0,j-1,s])
            derv2_v_y = dfdyc('2',j,v_old[0,j+1],v_old[0,j+0],v_old[0,j-1])
                
            # Load right-side corrector of old x derivatives values:
            derv2_v_x = dfdxc('2',0,v_old[1,j],v_old[0,j],v_old[1,j])
            
            # Calculate right-side value:
            v_right_help  = v_old[0,j] - dt*(xPRDCT[0,j,s]*derv1_v_y)
            
            term_help = - (curPRDCT[0,j,2]*bxPRDCT[0,j] - curPRDCT[0,j,0]*xPRDCT[0,j,5])
            c_coef = cz**2 + cx**2 + cs**2
            
            v_right_help -= dt*term_help/xPRDCT[0,j,0]
            v_right_help -= dt**2*((cx**2)*derv2_v_x + c_coef*derv2_v_y)
                
            if (gravLOG==True):
                delRHO_p1 = xPRDCT[0,j+1,0] - rhoINIT[0,j+1]
                delRHO_00 = xPRDCT[0,j+0,0] - rhoINIT[0,j+0]
                delRHO_m1 = xPRDCT[0,j-1,0] - rhoINIT[0,j-1]
                                        
                v_right_help -= dt*( gasR*tempINIT*dfdyc('1',j,delRHO_p1,delRHO_00,delRHO_m1)   \
                                   + (delRHO_00*surfGRAV*(sunR**2)/(sunR + yy[j])**2)         ) \
                                     /xPRDCT[0,j,0]
            else:
                derv_p = dfdyc('1',j,xPRDCT[0,j+1,6],xPRDCT[0,j+0,6],xPRDCT[0,j-1,6])
                v_right_help -= dt*derv_p/xPRDCT[0,j,0]
                
            # Load calculated right-side value into vRIGHT array:
            vRIGHT[0,j] = v_right_help
            
    # (3) Calculate the right-side corrector values for vx,vy,vz at the bottom boundary:
    if (s==1 or s==2):
        for i in range(mxp):
            vRIGHT[i,0] = 0.0  # Vx,Vy at bottom boundary is 0. (no converging flow at boundary)
    elif (s==3):
        if ((timeSIM+dt)<tACC):
            for i in range(mxp):
                vRIGHT[i,0] = ((timeSIM+dt)/tACC)*vzMAX*xx[i]*np.exp(0.5*(1.0-xx[i]**2))
        else:
            for i in range(mxp):
                vRIGHT[i,0] = vzMAX*xx[i]*np.exp(0.5*(1.0-xx[i]**2))
                
    # (4) Calculate the right-side corrector values for vx,vy,vz at the top and right-side boundaries:
    #       !!! These do not need to be loaded since they should all be 0 anyways. 
            
            
#
def calc_correctLEFT(s, vRIGHT, dt):
    # Solves the advanced velocities (left-side of corrector equations) semi-implicitly from the
    # explicitly calculated right-side (vRIGHT) using SciPy's sparse matrix methods.
    
    if (s==1):
        cDATA   = cxDATA
        cIND    = cxIND
    elif (s==2):
        cDATA   = cyDATA
        cIND    = cyIND
    elif (s==3):
        cDATA   = czDATA
        cIND    = czIND
    
    cSPARSE = csc_matrix( (cDATA, (cIND[0,:], cIND[1,:])) )

    # (1) Change the vRIGHT matrix into a 1D vector:
    vRIGHT_VECTOR = np.ravel(vRIGHT, order='C')

    # (2) Solve the matrix equation giving the advanced velocities: 

    cLU_DEC = splu(cSPARSE)
    vADVCD  = cLU_DEC.solve(vRIGHT_VECTOR)
    
    # (3) Load the newly obtained advanced velocities into the x variable array:
    for r in range(mxp*myp):
        ii = int(r/myp)
        jj = r - (ii*myp)
        
        x[ii,jj,s] = vADVCD[r]
    
#
def calc_innADVCD(s,velHALF,xOLD,xPRDCT,bxPRDCT,byPRDCT,dt):
    # Calculates the advanced time steps for the rest of the MHD variables using the velocity half-step 
    # values. 
    # velHALF indexing:    0: vx_advanced; 1: vy_advanced; 2: vz_advanced
    
    for i in range(1,mxp-1):
        for j in range(1,myp-1):
    
            # rho_advanced:
            if (s==0):
                fxp1 = xPRDCT[i+1,j,0]*velHALF[i+1,j,0]
                fx0  = xPRDCT[i+0,j,0]*velHALF[i+0,j,0]
                fxm1 = xPRDCT[i-1,j,0]*velHALF[i-1,j,0]
            
                fyp1 = xPRDCT[i,j+1,0]*velHALF[i,j+1,1]
                fy0  = xPRDCT[i,j+0,0]*velHALF[i,j+0,1]
                fym1 = xPRDCT[i,j-1,0]*velHALF[i,j-1,1]
            
                x[i,j,0] = xOLD[i,j,0] - dt*( dfdxc('1',i,fxp1,fx0,fxm1)  \
                                            + dfdyc('1',j,fyp1,fy0,fym1)  )
                                  
            # psi_advanced:
            elif (s==4):
                fxp1 = xPRDCT[i+1,j,4]
                fx0  = xPRDCT[i+0,j,4]
                fxm1 = xPRDCT[i-1,j,4]
            
                fyp1 = xPRDCT[i,j+1,4]
                fy0  = xPRDCT[i,j+0,4]
                fym1 = xPRDCT[i,j-1,4]
            
                x[i,j,4] = xOLD[i,j,4] - dt*( velHALF[i,j,0]*dfdxc('1',i,fxp1,fx0,fxm1) \
                                            + velHALF[i,j,1]*dfdyc('1',j,fyp1,fy0,fym1) )
                                             
            # Bz_advanced:
            elif (s==5):
                fxp1 = velHALF[i+1,j,2]*bxPRDCT[i+1,j] - velHALF[i+1,j,0]*xPRDCT[i+1,j,5]
                fx0  = velHALF[i+0,j,2]*bxPRDCT[i+0,j] - velHALF[i+0,j,0]*xPRDCT[i+0,j,5]
                fxm1 = velHALF[i-1,j,2]*bxPRDCT[i-1,j] - velHALF[i-1,j,0]*xPRDCT[i-1,j,5]
            
                fyp1 = velHALF[i,j+1,2]*byPRDCT[i,j+1] - velHALF[i,j+1,1]*xPRDCT[i,j+1,5]
                fy0  = velHALF[i,j+0,2]*byPRDCT[i,j+0] - velHALF[i,j+0,1]*xPRDCT[i,j+0,5]
                fym1 = velHALF[i,j-1,2]*byPRDCT[i,j-1] - velHALF[i,j-1,1]*xPRDCT[i,j-1,5]
            
                x[i,j,5] = xOLD[i,j,5] + dt*( dfdxc('1',i,fxp1,fx0,fxm1)  \
                                            + dfdyc('1',j,fyp1,fy0,fym1)  )
                                             
            # pressure_predct1:
            elif (s==6):
                x[i,j,6] = x[i,j,0]*gasR*tempINIT
            
            else:
                print("Error...too many iterations for calc_innADVCD function.")
                
                
#
def calc_outADVCD(s,velHALF,xOLD,xPRDCT,bxPRDCT,byPRDCT,dt): 
    # Calculates the advanced values for all boundary points.
    # Order: Left-side (open); Top (Open); Right-side (Open); Bottom (Fixed/Open)
    # velHALF indexing:    0: vx_advanced; 1: vy_advanced; 2: vz_advanced
    
    # (1): Left-side boundary ----------------------------
    for j in range(1,myp-1):
    
        # rho_advanced
        if (s==0):
            fxp1 = +velHALF[1,j,0]
            fx0  = +0.0
            fxm1 = -velHALF[1,j,0]
            
            fyp1 = xPRDCT[0,j+1,0]*velHALF[0,j+1,1]
            fy0  = xPRDCT[0,j+0,0]*velHALF[0,j+0,1]
            fym1 = xPRDCT[0,j-1,0]*velHALF[0,j-1,1]
            
            x[0,j,0] = xOLD[0,j,0] - dt*(                                   \
                            (dfdxc('1',0,fxp1,fx0,fxm1)*xPRDCT[0,j,0])      \
                           + dfdyc('1',j,fyp1,fy0,fym1)                     )
        
        # psi_advanced
        elif (s==4):
            fyp1 = xPRDCT[0,j+1,4]
            fy0  = xPRDCT[0,j+0,4]
            fym1 = xPRDCT[0,j-1,4]
            
            x[0,j,4] = xOLD[0,j,4] - dt*( velHALF[0,j,1]*dfdyc('1',j,fyp1,fy0,fym1) )

        # Bz_advanced
        elif (s==5):
            dervx_1 = dfdxc('1',0,velHALF[1,j,2],0.0,-velHALF[1,j,2])
            dervx_2 = dfdxc('1',0,velHALF[1,j,0],0.0,-velHALF[1,j,0])

            dervy_1 = dfdyc('1',j,xPRDCT[0,j+1,5],xPRDCT[0,j+0,5],xPRDCT[0,j-1,5])
            dervy_2 = dfdyc('1',j,velHALF[0,j+1,1],velHALF[0,j+0,1],velHALF[0,j-1,1])
            
            x[0,j,5] = xOLD[0,j,5] + dt*(    dervx_1*bxPRDCT[0,j]       \
                                         -   dervx_2*xPRDCT[0,j,5]      \
                                         -   dervy_1*velHALF[0,j,1]     \
                                         -   dervy_2*xPRDCT[0,j,5]      )

        # pressure_advanced
        elif (s==6):
            x[0,j,6] = x[0,j,0]*gasR*tempINIT
        
    # (2): Top boundary ----------------------------
    for i in range(0,mxp-1):
    
        # rho_advanced
        if (s==0):
            fym1 = x[i,myp-2,0] - xOLD[i,myp-2,0]
            fym2 = x[i,myp-3,0] - xOLD[i,myp-3,0]
        
            x[i,myp-1,0] = xOLD[i,myp-1,0] - (  (dymCOEF[myp-1,0,1]*fym1    \
                                               + dymCOEF[myp-1,0,2]*fym2)   \
                                                 /dymCOEF[myp-1,0,0]        )
        
        # psi_advanced
        elif (s==4):
            if (i==0):
                derv_term  = 0.0
            else:
                fxp1 = xPRDCT[i+1,myp-1,4]
                fx0  = xPRDCT[i+0,myp-1,4]
                fxm1 = xPRDCT[i-1,myp-1,4]
            
                derv_term = velHALF[i,myp-1,0] * dfdxc('1',i, fxp1, fx0, fxm1 )
            
            if (velHALF[i,myp-1,1] > 0.):
                fy0  = xPRDCT[i,myp-1,4]
                fym1 = xPRDCT[i,myp-2,4]
                fym2 = xPRDCT[i,myp-3,4]
                
                derv_term += velHALF[i,myp-1,1] * dfdym('1',myp-1, fy0, fym1, fym2 )
                
            x[i,myp-1,4] = xOLD[i,myp-1,4] - dt*derv_term

            
        # Bz_advanced
        elif (s==5):
            x[i,myp-1,5] = - ( (dymCOEF[myp-1,0,1]*x[i,myp-2,5]     \
                              + dymCOEF[myp-1,0,2]*x[i,myp-3,5])    \
                                /dymCOEF[myp-1,0,0] )

        # pressure_advanced
        elif (s==6):
            x[i,myp-1,6] = x[i,myp-1,0]*gasR*tempINIT
        
        
    # (3): Right-side boundary ----------------------------
    for j in range(1,myp):
    
        # rho_advanced
        if (s==0):
            fxm1 = x[mxp-2,j,0] - xOLD[mxp-2,j,0]
            fxm2 = x[mxp-3,j,0] - xOLD[mxp-3,j,0]
        
            x[mxp-1,j,0] = xOLD[mxp-1,j,0] - ( (dxmCOEF[mxp-1,0,1]*fxm1     \
                                              + dxmCOEF[mxp-1,0,2]*fxm2)    \
                                                /dxmCOEF[mxp-1,0,0] )
        
        # psi_advanced
        if (s==4):
            if (j==0):
                # j==0 value will be calculated in bottom boundary.
                pass
            elif (j==myp-1):
                derv_term = 0.0
                
                if (velHALF[mxp-1,j,1] > 0.):
                    fy0  = xPRDCT[mxp-1,j+0,4]
                    fym1 = xPRDCT[mxp-1,j-1,4]
                    fym2 = xPRDCT[mxp-1,j-2,4]
            
                    derv_term += velHALF[mxp-1,j,1] * dfdym('1',j, fy0, fym1, fym2 )
            
                if (velHALF[mxp-1,j,0] > 0.):
                    fx0  = xPRDCT[mxp-1,j,4]
                    fxm1 = xPRDCT[mxp-2,j,4]
                    fxm2 = xPRDCT[mxp-3,j,4]
                
                    derv_term += velHALF[mxp-1,j,0] * dfdxm('1',mxp-1, fx0, fxm1, fxm2 )
            else:
                fyp1 = xPRDCT[mxp-1,j+1,4]
                fy0  = xPRDCT[mxp-1,j+0,4]
                fym1 = xPRDCT[mxp-1,j-1,4]
            
                derv_term = velHALF[mxp-1,j,1] * dfdyc('1',j, fyp1, fy0, fym1 )
            
                if (velHALF[mxp-1,j,0] > 0.):
                    fx0  = xPRDCT[mxp-1,j,4]
                    fxm1 = xPRDCT[mxp-2,j,4]
                    fxm2 = xPRDCT[mxp-3,j,4]
                
                    derv_term += velHALF[mxp-1,j,0] * dfdxm('1',mxp-1, fx0, fxm1, fxm2 )
                
            x[mxp-1,j,4] = xOLD[mxp-1,j,4] - dt*derv_term

        # Bz_advanced
        elif (s==5):
            x[mxp-1,j,5] = - ( (dxmCOEF[mxp-1,0,1]*x[mxp-2,j,5]     \
                              + dxmCOEF[mxp-1,0,2]*x[mxp-3,j,5])    \
                                /dxmCOEF[mxp-1,0,0] )

        # pressure_advanced
        elif (s==6):
            x[mxp-1,j,6] = x[mxp-1,j,0]*gasR*tempINIT
        
        
    # (4): Bottom boundary ----------------------------
    for i in range(0,mxp):
        
        # rho_advanced
        if (s==0):
            x[i,0,0] = xOLD[i,0,0] - dt*(xPRDCT[i,0,0]*dfdyp('1',0,velHALF[i,2,1],velHALF[i,1,1],velHALF[i,0,1]))
        
        # psi_advanced
        elif (s==4):
            x[i,0,4] = xOLD[i,0,4]

        # Bz_advanced
        elif(s==5):
            if (i==0):
                derv_xa = dfdxc('1',0,velHALF[1,0,2],velHALF[0,0,2],-velHALF[1,0,2])
                derv_ya = dfdyp('1',0,velHALF[0,2,1],velHALF[0,1,1],velHALF[0,0,1])
            
                x[i,0,5] = xOLD[i,0,5] + dt*(  derv_xa*bxPRDCT[0,0]   \
                                             - derv_ya*xPRDCT[0,0,5]  )

            elif (i==mxp-1):
                derv_xa = dfdxm('1',i,bxPRDCT[i+0,0],bxPRDCT[i-1,0],bxPRDCT[i-2,0])
                derv_xb = dfdxm('1',i,velHALF[i+0,0,2],velHALF[i-1,0,2],velHALF[i-2,0,2])
                
                derv_ya = dfdyp('1',0,byPRDCT[i,2],byPRDCT[i,1],byPRDCT[i,0])
                derv_yb = dfdyp('1',0,velHALF[i,2,2],velHALF[i,1,2],velHALF[i,0,2])
                derv_yc = dfdyp('1',0,velHALF[i,2,1],velHALF[i,1,1],velHALF[i,0,1])
                
                x[i,0,5] = xOLD[i,0,5] + dt*(   derv_xa*velHALF[i,0,2]  \
                                              + derv_xb*bxPRDCT[i,0]    \
                                              + derv_ya*velHALF[i,0,2]  \
                                              + derv_yb*byPRDCT[i,0]    \
                                              - derv_yc*xPRDCT[i,0,5]   )
                                               
            else:
                derv_xa = dfdxc('1',i,bxPRDCT[i+1,0],bxPRDCT[i+0,0],bxPRDCT[i-1,0])
                derv_xb = dfdxc('1',i,velHALF[i+1,0,2],velHALF[i+0,0,2],velHALF[i-1,0,2])
            
                derv_ya = dfdyp('1',0,byPRDCT[i,2],byPRDCT[i,1],byPRDCT[i,0])
                derv_yb = dfdyp('1',0,velHALF[i,2,2],velHALF[i,1,2],velHALF[i,0,2])
                derv_yc = dfdyp('1',0,velHALF[i,2,1],velHALF[i,1,1],velHALF[i,0,1])
        
                x[i,0,5] = xOLD[i,0,5] + dt*(   derv_xa*velHALF[i,0,2]     \
                                            +   derv_xb*bxPRDCT[i,0]       \
                                            +   derv_ya*velHALF[i,0,2]     \
                                            +   derv_yb*byPRDCT[i,0]       \
                                            -   derv_yc*xPRDCT[i,0,5]      )
        
        # pressure_advanced
        elif (s==6):
            x[i,0,6] = x[i,0,0]*gasR*tempINIT
 

#
def record():
    global x, bx, by, cur
    # Records all simulated variables in a byte file. 

    xALL = np.zeros((mxp,myp,12), dtype=np.double, order='C')
    
    for s in range(7):
        xALL[:,:,s] = x[:,:,s]
        
    xALL[:,:,7]  = bx[:,:]
    xALL[:,:,8]  = by[:,:]
    xALL[:,:,9]  = cur[:,:,0]
    xALL[:,:,10] = cur[:,:,1]
    xALL[:,:,11] = cur[:,:,2]
    
    
    filename = "V1.12_n" + str(nstep)
    recordFILE = open(filename,"wb")
    np.save(recordFILE,xALL)
    recordFILE.close()
   
   
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
            
    xxxDOUBLE = np.zeros(int(2.*mxp))
    for i in range(mxp-1):
        xxxDOUBLE[i] = - xx[-(i+1)]
    for i in range(0,mxp+1):
        if (i==mxp):
            dx = xx[i-1] - xx[i-2]
            xxxDOUBLE[i+mxp-1] = + xx[i-1] + dx
        else:
            xxxDOUBLE[i+mxp-1] = + xx[i]
    
    # plot 1 -------------------------
    plt.subplot(231, aspect='equal')
    plt.suptitle("V1.12 | vis1,vis2=1/150,1/20 | shear=%.5f | cfl=%.3f" %(shearMAX,cfl), fontsize=15)
    
    xDOUBLE = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE[i,j] = x[i-(mxp-1),j,1]
            else:
                xDOUBLE[i,j] = -x[(mxp-1)-i,j,1]
                

    plt.pcolormesh(xxxDOUBLE,yyy,xDOUBLE.T, cmap='rainbow', rasterized=True)

    #plt.xlabel(r'$x}$', fontsize=f_size+0.2)
    plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Vx at Step: {a}" .format(a=nstep))



    # plot 2 -------------------------
    plt.subplot(232, aspect='equal')
    
    xDOUBLE = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE[i,j] = +x[i-(mxp-1),j,2]
            else:
                xDOUBLE[i,j] = +x[(mxp-1)-i,j,2]

    plt.pcolormesh(xxxDOUBLE,yyy,xDOUBLE.T, cmap='rainbow', rasterized=True)

    #plt.xlabel(r'$x$', fontsize=f_size+0.2)
    #plt.ylabel(r'$y$', fontsize=f_size+0.2)  
    cb = plt.colorbar()
    plt.title("Vy at Step: {a}" .format(a=nstep))


    # plot 3 -------------------------
    plt.subplot(233, aspect='equal')
    
    xDOUBLE = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE[i,j] = +x[i-(mxp-1),j,3]
            else:
                xDOUBLE[i,j] = -x[(mxp-1)-i,j,3]

    plt.pcolormesh(xxxDOUBLE,yyy,xDOUBLE.T, cmap='rainbow', rasterized=True)

#    plt.xlabel(r'$x$', fontsize=f_size+0.2)
#    plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Vz at Step: {a}" .format(a=nstep))


    # plot 4 -------------------------
    plt.subplot(234, aspect='equal')
    
    xDOUBLE = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE[i,j] = +dRho[i-(mxp-1),j]
            else:
                xDOUBLE[i,j] = +dRho[(mxp-1)-i,j]

    plt.pcolormesh(xxxDOUBLE,yyy,xDOUBLE.T, cmap='rainbow', rasterized=True)

    plt.xlabel(r'$x$', fontsize=f_size+0.2)
    plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("dRho at Step: {a}" .format(a=nstep))
    
    # plot 5 -------------------------
    plt.subplot(235, aspect='equal')
    
    xDOUBLE = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE[i,j] = +x[i-(mxp-1),j,5]
            else:
                xDOUBLE[i,j] = +x[(mxp-1)-i,j,5]

    plt.pcolormesh(xxxDOUBLE,yyy,xDOUBLE.T, cmap='rainbow', rasterized=True)

    plt.xlabel(r'$x$', fontsize=f_size+0.2)
    #plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Bz at Step: {a}" .format(a=nstep))

    
    # plot 6 -------------------------
    plt.subplot(236, aspect='equal')
    
    levels = 25
    xDOUBLE1 = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE1[i,j] = +psiINIT[i-(mxp-1),j]
            else:
                xDOUBLE1[i,j] = +psiINIT[(mxp-1)-i,j]
                
    xDOUBLE2 = np.zeros((int(2*mxp-1),myp))
    for i in range(int(2*mxp-1)):
        for j in range(myp):
            if (i>(mxp-1)):
                xDOUBLE2[i,j] = +x[i-(mxp-1),j,4]
            else:
                xDOUBLE2[i,j] = +x[(mxp-1)-i,j,4]
    
    
    plt.contour(xxxDOUBLE[0:-1],yy,xDOUBLE1.T, levels, colors='gray', linewidths=0.6,linestyles=':')  # no color
    plt.contour(xxxDOUBLE[0:-1],yy,xDOUBLE2.T, levels, colors='k', linewidths=0.6)                   # no color
    
    plt.pcolormesh(xxxDOUBLE,yyy,xDOUBLE2.T, cmap='brg', rasterized=True)

    plt.xlabel(r'$x$', fontsize=f_size+0.2)
    #plt.ylabel(r'$y$', fontsize=f_size+0.2)
    cb = plt.colorbar()
    plt.title("Psi at Step: {a}" .format(a=nstep))


    filename = "V1.12_n" + str(nstep)
    ##plt.tight_layout(h_pad=1.75)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.10, hspace=0.20)
    plt.savefig('%s.eps' %filename, format='eps', bbox_inches='tight', dpi=400)
#    plt.show()
   

   


# ------
main()


