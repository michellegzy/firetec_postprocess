import numpy as np
import math

def consumption_and_reaction_heat(tkb,O2,rho,rhof,rhof0,rhow,temps): 
    # cts
    rnfuel      = 0.4552
    rno         = 0.5448
    tcrit       = 600.
    tfstep      = 310.
    cfhydro     = 0.9
    cfchar      = 0.09
    hydroThresh = 0.4
    sc          = 0.1
    c1          = 0.5
    c2          = 0.0079
    c3          = 1.

    # turb mixing 
    fcorr = 0.5
    rkctemp = 0.2*tkb*fcorr/rho
    sigmac = sc*0.5*math.sqrt(rkctemp)

    # flame heat, reaction extent, and length correlations
    if(temps<tfstep):
        psif = 0.
    elif(temps > (2.*(tcrit-tfstep)+tfstep)):
        psif = 1.
    else:
        psif = c1*(c3+math.erf(c2*(temps-tcrit)))
    percHydroRemaining = max(0.,(rhof-hydroThresh*rhof0)/(rhof0*(1.-hydroThresh)))
    cf = cfhydro*percHydroRemaining+cfchar*(1.-percHydroRemaining)

    # fuel consumption
    slambdaof = rhof*O2/(rhof/rnfuel+O2/rno)**2.
    frhof = rnfuel*cf*rhof*O2*sigmac*psif*slambdaof/(100.*0.0005**2.)

    return frhof
