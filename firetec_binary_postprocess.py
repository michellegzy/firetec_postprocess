#!/usr/bin/python
# from pyevtk.hl import gridToVTK
import numpy as np
import struct
import os.path
import sys
from h5py import File
import matplotlib.pylab as plt 
import pandas as pd 
import math 
# import os

# --- define input and output directories ---
cwd           = os.getcwd()
indir         = cwd           # pathfile to comp.outs 
readfilename  = '/comp.out.'  # name of comp.out files
outdir        = './postprocessing/'          # output directory 
outname       = 'vtk_output.' # output name to name vtk
gridlist_pf   = cwd            # pathfile to the gridlist 
csv_outpath = './hflux_data.csv' #~/Desktop/het_fuels_proj/heat_flux_data.csv'

# --- define time, fuel, and firerun parameters ---
initial = 1000 #4000 # time of ignition
final = 120001
incr  = 1000
nfuel = 2 # change to match your number of fuel types
fields_to_write = [] # define what fields to write OR leave empty for full FIRETEC output

# --- define domain size and parameters ---
Nx = 200 
Ny = 100 
Nz = 41
Nzfuel = 1
dx = 2.0
dy = 2.0
dz = 15.0
aa1 = 0.1
f0 = 0.0
stretch = 2 #0=no vertical stretch, 2=cubic, 1=not avaible
topofile = '' #topo input here, leave empty for flat 

# ------- cts and vars ----------- 
h = 34.21 # [w/m2K] 
T_inf = 300 # K
epsilon = 0.96 # grass emissivity 
sigma = 5.67 * 10 ** -8 #[w/m2K4]

#======================= DEFINE FUNCTIONS =========================                                                                                            

def formOutputList(pf,fields_to_write,nfuel):
    fn = pf+'/gridlist'
    if os.path.exists(fn):
        gl = open(fn)
        lines = gl.readlines()
        
        # Gas velocities and temperature 
        gas_field_names = ["u", "v", "w", "theta"]

        # Turbulence parameters
        kab = [s for s in lines if "iturb" in s]
        if (any('2' in s for s in kab)):
            gas_field_names.extend(["ka","kb"])
        
        # Emissions species
        emit = [s for s in lines if "iemissions" in s]
        if (any('1' in s for s in emit)) | (any('2' in s for s in emit)):
            nemit_ch = [s for s in lines if "nEmit" in s]
            nemit = [int(s) for s in nemit_ch[0].split() if s.isdigit()][0]
            if(nemit != 0):
                spemit_ch = [s for s in lines if "Species" in s and "SpeciesAero" not in s]
                for i in range(nemit):
                    gas_field_names.extend([spemit_ch[0].split()[2+i]])
        fire = [s for s in lines if "ifire" in s]
        if (any('1' in s for s in fire)):
            if (any('1' in s for s in emit)) or (any('2' in s for s in emit)):
                if (nemit !=0):
                    if not (any(char == "O2" for char in spemit_ch[0].split())):
                        gas_field_names.extend([''])
                        gas_field_names[-nemit-1]="O2"
                        for i in range(nemit):
                            gas_field_names[-nemit+i]=spemit_ch[0].split()[2+i]
                else:
                    gas_field_names.extend(["O2"])
            else:
                gas_field_names.extend(["O2"])
#        if (any('1' in s for s in emit)) | (any('2' in s for s in emit)):
#            naero_ch = [s for s in lines if "nAero" in s]
#            nmaero_ch = [s for s in lines if "nMAero" in s]
#            naero = [int(s) for s in naero_ch[0].split() if s.isdigit()][0]
 #           nmaero = [int(s) for s in nmaero_ch[0].split() if s.isdigit()][0]
  #          if(naero != 0):
   #             spaero_ch = [s for s in lines if "SpeciesAero" in s]
    #            for i in range(naero):
     #               for j in range(nmaero):
      #                  gas_field_names.extend([spaero_ch[0].split()[2+i]+str(j)])
        if any('3' in s for s in emit): 
            gas_field_names.extend(['M0','M1'])

        # Mixture Fraction
        mixfrac = [s for s in lines if "inonlocal" in s]
        if any('1' in s for s in mixfrac):
            gas_field_names.extend(['mixfrac'])

        # Gas Density
        gas_field_names.append('density')

        # Fuel Arrays
        ifuel = [s for s in lines if "nfuel" in s] # look for string 'nfuel' in GL
        if (any('=' in s for s in ifuel)): # if there is an = sign: 
            nfuel = int(ifuel[0].split()[2]) # convert the third index to an integer (exp. nfuel = 2, converts '2' to int)
        else: 
            nfuel = 1
        if nfuel == 2:
            fuel_field_names = ['rhoFuel_1', 'rhoFuel_2', 'rho_water1', 'rho_water2', 'sies1', 'sies2', 'psi_wmax1', 'psi_wmax2'] # for n=2
        if nfuel==1:
            fuel_field_names = ['rhoFuel', 'rho_water', 'sies', 'psi_wmax']
        #fuel_field_names = [] # initialize
        #for i in range(nfuel):
        #    fuel_field_names.extend(['rhoFuel'+str(i+1), 'rhoWater'+str(i+1)])
        #    if (any('1' in s for s in fire)):
        #        fuel_field_names.extend(['sies'+str(i+1), 'psiwmax'+str(i+1)])
        gl.close()

        if len(fields_to_write)==0:
            fields_to_write = gas_field_names+fuel_field_names

        # fields that need to be divided by density -- same for wind and firerun    
        div_by_dens = gas_field_names.copy()
        div_by_dens.remove('density')
    else:
        print ('Not a valid gridlist pathfile! Exiting now...')
        quit()
    return gas_field_names,fuel_field_names,div_by_dens,fields_to_write

def readfield(infile, Nx, Ny, Nz):
    raw_data = infile.read(Nx * Ny * Nz * 4)  # read binary data
    array = np.frombuffer(raw_data, dtype=np.float32)  # converp to np array

    #print(f"Expected size: {Nx*Ny*Nz}, Read size: {array.size}") # make sure array sizes are as expected

    if array.size != Nx * Ny * Nz:
        raise ValueError(f"File does not contain enough data: Expected {Nx*Ny*Nz}, got {array.size}") # perhaps they are not!!!

    return array.reshape((Nx, Ny, Nz), order='F')  # reshape with fortran order
    #return np.frombuffer(infile.read(Nx*Ny*Nz*4), 'f').reshape((Nx,Ny,Nz),order='F') # single line from original function

def read_fields(fname, Nx, Ny, Nz, Nzfuel, number, gas_field_names, fuel_field_names, div_by_dens): #debugging function
    outputs = {}

    file_path = fname + str(number)  # construct filename
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as infile:
        #print(f"Reading file: {file_path}")

        for ii in range(len(gas_field_names)):
            infile.read(4)  # skip 4 bytes for header
            outputs[gas_field_names[ii]] = readfield(infile, Nx, Ny, Nz)
            infile.read(4)  # skip 4 bytes for footer (fortran record markers)

        for ii in range(len(fuel_field_names)):
            infile.read(4)
            FuelTemp = readfield(infile, Nx, Ny, Nzfuel)
            outputs[fuel_field_names[ii]] = np.zeros((Nx, Ny, Nz)) # ensure the array is properly shaped
            outputs[fuel_field_names[ii]][:, :, :Nzfuel] = FuelTemp
            infile.read(4)
            
        for ii in range(len(div_by_dens)):  # Loop over each field that needs density normalization
            the_field = div_by_dens[ii]  # Get the field name
            outputs[the_field] = np.divide(outputs[the_field], outputs["density"])  # Element-wise division
            
    return outputs

def select_data(pD, fields_to_write):
  output = {}
  for field in fields_to_write:
      output[field] = pD[field]
  return output

def zheight(ZI):
# generates array of cell heights from z-index array                                                                                
  Z = np.copy(ZI, order='K')
  ZItemp = Z[0,0,:]
  ZItemp[0] = ZItemp[0] * 2
  for ii in range(1,len(ZItemp)):
          ZItemp[ii] = (ZItemp[ii] - sum(ZItemp[:ii]))*2
  for ii in range(len(ZItemp)):
          Z[:,:,ii] = ZItemp[ii]
  return Z

def metrics(topofile, Nx, Ny, Nz, dx, dy, dz, a1, f0, Stretch):
  # --- read topo file if present ---
  if os.path.isfile(topofile):
    #topo = numpy.zeros((Nx,Ny))
    f = open(topofile, 'rb')
    f.seek(4)
    topo=np.frombuffer(f.read(Nx*Ny*4),'f').reshape((Nx,Ny), order = 'F')
    f.close()

  # --- build base grid ---
  x = np.zeros((Nx))
  y = np.zeros((Ny))
  z = np.zeros((Nz))
  zedge = np.zeros((Nz+1))
  XI = np.zeros((Nx,Ny,Nz))
  YI = np.zeros((Nx,Ny,Nz))
  ZI = np.zeros((Nx,Ny,Nz))
  for i in range(Nx):
      x[i] = i*dx - 0.5*Nx*dx
  for j in range(Ny):
      y[j] = j*dy - 0.5*Ny*dy
  for k in range(Nz):
      z[k] = k*dz + 0.5*dz
      zedge[k] = k*dz

  zedge[Nz] = Nz*dz
  # --- using no stretching ---
  if Stretch == 0:
      print('not using stretching')
      for i in range(Nx):
          for j in range(Ny):
              for k in range(Nz):
                  XI[i,j,k] = x[i]
                  YI[i,j,k] = y[j]
                  ZI[i,j,k] = z[k]    

  # --- using hyperbolic tangent stretching ---
  if Stretch == 1:
      print('using hyperbolic tangent stretching')
      print('this part does not work yet! exiting!')
      sys.exit()

  # --- using cubic polynomial stretching ---
  if Stretch == 2:
      print('using cubic polynomial stretching')
  # --- set cubic polynomial 2nd and 3rd term coefficients ---
      a2 = f0*(1.0-a1)/zedge[Nz]
      a3 = (1.0-a2*zedge[Nz]-a1)/(zedge[Nz]**2.0)
      for i in range(Nx):
          for j in range(Ny):
              for k in range(Nz):
                  XI[i,j,k] = x[i]
                  YI[i,j,k] = y[j]
                  ZI[i,j,k] = (a3*(z[k]**3.0)+a2*(z[k]**2.0)+a1*z[k])*(zedge[Nz]-zedge[0])/zedge[Nz]+zedge[0]

      if os.path.isfile(topofile):
          print("Modifying coordinate to be terrain following!")
          for i in range(Nx):
              for j in range(Ny):
                  for k in range(Nz):
                      ZI[i,j,k] = ZI[i,j,k]*(zedge[Nz]-topo[i,j])/zedge[Nz] + topo[i,j]
  Z = zheight(ZI)
  volume = np.multiply(dx,dy,Z)
  return XI, YI, ZI, volume

def total_fuel_consumption(comp_out_initial, comp_out_final, Nx, Ny, Nz, Nzfuel, gas_field_names, fuel_field_names, div_by_dens):
    """
    reads 'rhoFuel_1' and 'rhoFuel_2' from the given fortran binary files,
    sums them to get total fuel density at initial and final steps,
    and computes the correct ratio (final/initial).
    """
    # check both files exist
    if not os.path.exists(comp_out_initial) or not os.path.exists(comp_out_final):
        raise FileNotFoundError("one or both of the specified 'comp.out' not found in cwd/filepath.")

    # extract the base filename without the number
    fname_initial = ".".join(comp_out_initial.split(".")[:-1]) + "."
    fname_final = ".".join(comp_out_final.split(".")[:-1]) + "."

    # extract time step numbers from the filenames
    number_initial = int(comp_out_initial.split('.')[-1])
    number_final = int(comp_out_final.split('.')[-1])

    # read the initial fuel data
    initial_data = read_fields(fname_initial, Nx, Ny, Nz, Nzfuel, number_initial, gas_field_names, fuel_field_names, div_by_dens)
    if nfuel == 1: 
        rho_fuel_initial = initial_data["rhoFuel"]
        rho_fuel_tot_initial = np.sum(initial_data["rhoFuel"]) 
    if nfuel == 2:
        rho_fuel_initial1 = initial_data["rhoFuel_1"] 
        rho_fuel_initial2 = initial_data["rhoFuel_2"] 
        rho_fuel_tot_initial = np.sum(initial_data["rhoFuel_1"]) + np.sum(initial_data["rhoFuel_2"]) 

    # read the final fuel data
    final_data = read_fields(fname_final, Nx, Ny, Nz, Nzfuel, number_final, gas_field_names, fuel_field_names, div_by_dens)
    if nfuel == 1:
        rho_fuel_tot_final = np.sum(final_data["rhoFuel"]) # + np.sum(final_data["rhoFuel_2"])
    if nfuel == 2:
        rho_fuel_tot_final = np.sum(final_data["rhoFuel_1"]) + np.sum(final_data["rhoFuel_2"])

    # compute the fuel consumption (final / initial)
    fuel_ratio = rho_fuel_tot_final / rho_fuel_tot_initial if rho_fuel_tot_initial != 0 else np.nan # % fuel remaining at EOS
    consumption = 1 - fuel_ratio # % consumed 
    initial_fuel_density = rho_fuel_tot_initial / (Nx * Ny) # calc single cell initial fuel density for fire spread rate, assume uniform loading
    
    return rho_fuel_initial1, rho_fuel_initial2, rho_fuel_tot_initial, rho_fuel_tot_final, initial_fuel_density, consumption

def consumption_and_reaction_heat(tkb, O2, rho_fuel_initial1, rho_fuel_initial2, rhoFuel_1, rhoFuel_2, rho_water1, rho_water2, ftemp1, ftemp2): 
    """
    Compute fuel consumption rate and chemical heat release.
    Works on arrays (entire grid at z=0).
    
    Returns:
    - frhof1, frhof2: fuel consumption rates [kg/m³/s]
    - reactFuelGas1, reactFuelGas2: heat release rate to gas [W/m³]
    """
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
    hf          = 8913.48e3   # heat of reaction for simple wood (J/kg of products)
    thetag      = 0.75        # fraction of energy to gas 

    # turbulent mixing
    fcorr = 0.5
    rkctemp = 0.2 * tkb * fcorr
    sigmac = sc * 0.5 * np.sqrt(np.maximum(rkctemp, 0))  # avoid negative vals

    # flame heat, reaction extent 
    psif1 = np.where(ftemp1 < tfstep, 0.0,
                     np.where(ftemp1 > (2.*(tcrit - tfstep) + tfstep), 1.0,
                              c1 * (c3 + np.vectorize(math.erf)(c2 * (ftemp1 - tcrit)))))
    
    psif2 = np.where(ftemp2 < tfstep, 0.0,
                     np.where(ftemp2 > (2.*(tcrit - tfstep) + tfstep), 1.0,
                              c1 * (c3 + np.vectorize(math.erf)(c2 * (ftemp2 - tcrit)))))

    percHydroRemaining1 = np.maximum(0., (rhoFuel_1 - hydroThresh * rho_fuel_initial1) / 
                                     (rho_fuel_initial1 * (1. - hydroThresh) + 1e-10))
    percHydroRemaining2 = np.maximum(0., (rhoFuel_2 - hydroThresh * rho_fuel_initial2) / 
                                     (rho_fuel_initial2 * (1. - hydroThresh) + 1e-10))
    
    cf1 = cfhydro * percHydroRemaining1 + cfchar * (1. - percHydroRemaining1) 
    cf2 = cfhydro * percHydroRemaining2 + cfchar * (1. - percHydroRemaining2) 

    # fuel consumption rate
    slambdaof1 = rhoFuel_1 * O2 / (rhoFuel_1 / rnfuel + O2 / rno + 1e-10)**2.
    slambdaof2 = rhoFuel_2 * O2 / (rhoFuel_2 / rnfuel + O2 / rno + 1e-10)**2.

    frhof1 = rnfuel * cf1 * rhoFuel_1 * O2 * sigmac * psif1 * slambdaof1 / (100. * 0.0005**2.) 
    frhof2 = rnfuel * cf2 * rhoFuel_2 * O2 * sigmac * psif2 * slambdaof2 / (100. * 0.0005**2.) 
    
    # energy release to gas 
    hydroFactor1 = np.exp(-rno / rnfuel * psif1 * rhoFuel_1 / (O2 + 1e-10))
    hydroFactor2 = np.exp(-rno / rnfuel * psif2 * rhoFuel_2 / (O2 + 1e-10))

    thetaSolid1 = percHydroRemaining1 * (1. - thetag) * hydroFactor1 + (1. - percHydroRemaining1) * thetag 
    thetaSolid2 = percHydroRemaining2 * (1. - thetag) * hydroFactor2 + (1. - percHydroRemaining2) * thetag

    reactFuelGas1 = (1. - thetaSolid1) * hf * frhof1  # [W/m³]
    reactFuelGas2 = (1. - thetaSolid2) * hf * frhof2  # [W/m³]

    return frhof1, frhof2, reactFuelGas1, reactFuelGas2

def fire_spread(point_data, Nx, Ny, dx, initial_fuel_density, time, previous_position_x, previous_time):
    """
    computes fire spread rate at a single timestep.

    Parameters:
    - point_data: dictionary containing fuel data at the current timestep.
    - Nx, Ny: grid dimensions.
    - dx: grid cell size (meters).
    - initial_fuel_density: single value of initial fuel density per cell.
    - time: current timestep.
    - previous_position_x: fire front position from the last timestep.
    - previous_time: previous timestep.

    Returns:
    - fire_front_x: fire front position (meters).
    - spread_rate: fire spread rate (m/s) for this timestep.
    """

    # compute total fuel density per cell
    if nfuel==1: 
        total_fuel = point_data["rhoFuel"] 
    if nfuel==2:    
        total_fuel = point_data["rhoFuel_1"] + point_data["rhoFuel_2"] 

    # find fire front: first x location where fuel < 95% of initial density
    fire_front_x = None

    for x in range(Nx-1,-1,-1):
        if np.any(total_fuel[x, :, 0] < 0.95 * initial_fuel_density): 
            fire_front_x = (x + 1) * dx  # track fire front at leading edge of burned cell, then convert index to meters 
            
            # current_cell_theta = point_data["theta"][x, :, 0].mean()  # avg theta across all y for this x
            # previous_cell_theta = point_data["theta"][x-1, :, 0].mean() if x > 0 else None  # average theta for previous x, if it exists            
            # print(f'fire front at x = {x}')
            # print(f'theta at fire cell: {current_cell_theta:.2f}')
            # if previous_cell_theta is not None:
            #     print(f'theta at previous cell: {previous_cell_theta:.2f}')
            # else:
            #     print('theta at previous cell: N/A')
            break  


    spread_rate = None
    if fire_front_x is not None and previous_position_x is not None:
        delta_x = fire_front_x - previous_position_x
        delta_t = time - previous_time
        spread_rate = delta_x / delta_t 

    return fire_front_x, spread_rate

def heat_flux(point_data, Nx, Ny, cp_solid1, cp_solid2, nfuel, rho_fuel_initial1, rho_fuel_initial2):
    """
    Compute total heat flux for z=0 across the grid.
    Includes convective, radiative, and chemical reaction heat.
    
    Returns:
    - qdub_total: total heat flux [W]
    - ftemp1, ftemp2: fuel temperatures (for tracking)
    """
    qdub_total = 0.0
    ftemp1_out = None
    ftemp2_out = None

    if nfuel == 1:
        # extract 2-D arrays at z=0
        sies = point_data['sies'][:, :, 0]
        rhoFuel = point_data['rhoFuel'][:, :, 0]
        rho_water = point_data['rho_water'][:, :, 0]
        
        ftemp = sies / cp_solid1
        ftemp = np.maximum(ftemp, 300.0) 
        ftemp1_out = ftemp

        # mask where flames are active
        mask = ftemp >= 315.0

        # convective and radiative heat flux
        q_conv = h * (ftemp - T_inf)
        q_rad  = epsilon * sigma * (ftemp**4 - T_inf**4)
        
        # get turbulence and O2
        tkb = point_data.get('kb', np.zeros_like(ftemp))[:, :, 0]
        O2 = point_data.get('O2', 0.23 * np.ones_like(ftemp))[:, :, 0]
        
        # reaction heat
        _, _, reactFuelGas, _ = consumption_and_reaction_heat(
            tkb, O2, rho_fuel_initial1, 0, 
            rhoFuel, np.zeros_like(rhoFuel),
            rho_water, np.zeros_like(rho_water),
            ftemp, np.zeros_like(ftemp)
        )
        
        # total heat release (sum over active cells)
        qdub_total = np.sum((q_conv + q_rad + reactFuelGas * 0.7)[mask]) # correct rFG term for units--multiplied by fuel height 

    elif nfuel == 2:
        # extract 2-D arrays at z=0
        sies1 = point_data['sies1'][:, :, 0]
        sies2 = point_data['sies2'][:, :, 0]
        rhoFuel_1 = point_data['rhoFuel_1'][:, :, 0]
        rhoFuel_2 = point_data['rhoFuel_2'][:, :, 0]
        rho_water1 = point_data['rho_water1'][:, :, 0]
        rho_water2 = point_data['rho_water2'][:, :, 0]

        ftemp1 = np.maximum(sies1 / cp_solid1, 300.0)
        ftemp2 = np.maximum(sies2 / cp_solid2, 300.0)
        ftemp1_out = ftemp1
        ftemp2_out = ftemp2

        mask1 = ftemp1 >= 315.0
        mask2 = ftemp2 >= 315.0

        # heat flux
        q_conv1 = h * (ftemp1 - T_inf)
        q_rad1  = epsilon * sigma * (ftemp1**4 - T_inf**4)
        q_conv2 = h * (ftemp2 - T_inf)
        q_rad2  = epsilon * sigma * (ftemp2**4 - T_inf**4)

        tkb = point_data.get('kb', np.zeros_like(ftemp1))[:, :, 0]
        O2 = point_data.get('O2', 0.23 * np.ones_like(ftemp1))[:, :, 0]
        
        # reaction heat
        _, _, reactFuelGas1, reactFuelGas2 = consumption_and_reaction_heat(
            tkb, O2, rho_fuel_initial1, rho_fuel_initial2,
            rhoFuel_1, rhoFuel_2,
            rho_water1, rho_water2,
            ftemp1, ftemp2
        )
        
        # total heat release
        qdub_total = (np.sum((q_conv1 + q_rad1 + reactFuelGas1 * 0.7)[mask1]) + 
                      np.sum((q_conv2 + q_rad2 + reactFuelGas2)[mask2]))

    else:
        raise ValueError("nfuel must be 1 or 2")

    return qdub_total, ftemp1_out, ftemp2_out

def flame_depth(point_data, Nx, Ny, fire_front_x, dx):
    """
    Detects flame presence across the x-y grid at a given z-slice.
    Calculates flame depth from fire front.

    Parameters:
    - point_data (dict): dictionary of field data (including 'O2' and 'theta').
    - Nx, Ny (int): grid dimensions.
    - fire_front_x (float): x-position of the fire front in meters.
    - dx (float): grid cell size in meters.

    Returns:
    - flame_map (2D np.array): boolean map of flame presence (shape: Nx x Ny).
    - flame_count (int): number of cells where flames are detected.
    - max_flame_depth (float): maximum flame depth in meters.
    """
    flame_map = np.zeros((Nx, Ny), dtype=bool)
    flame_count = 0
    max_flame_depth = 0

    # convert fire_front_x from meters to grid index for looping 
    fire_front_index = int(fire_front_x / dx)

    for y in range(Ny): 
        for x in range(fire_front_index, -1, -1):  # start from fire front, move backwards
            # if point_data["O2"][x, y, 0] <= 0.22 and point_data["theta"][x, y, 0] >= 325: 
            if point_data["theta"][x,y,0] >= 325: 
                flame_map[x, y] = True
                flame_count += 1
                current_depth = (fire_front_index - x + 1) * dx # +1 to include the current cell
                max_flame_depth = max(max_flame_depth, current_depth)
            else:
                break  # stop counting when we hit a non-flame cell
    
    return flame_map, flame_count, max_flame_depth

def store_csv(qdub, simulation_name, output_csv): 
    """
    stores desired array as csv.
    
    Parameters:
    - qdub (change to desired corresponding array/name): list or numpy array of values for each timestep.
    - simulation_name: string identifier for the simulation (used as the column header).
    - output_csv: file path to store data.
    """

    # convert `qdub` into a Pandas Series
    qdub_series = pd.Series(qdub, name=simulation_name)

    # check if csv already exists
    #if not os.path.exists(output_csv):
        #os.make(output_csv)
    if os.path.exists(csv_outpath): #output_csv):
        # if file exists, read into a dataframe
        df = pd.read_csv(csv_outpath, index_col=0) #output_csv, index_col=0)

        # ensure the new qdub has the same length as previous simulations
        if len(qdub_series) != len(df):
            print("Warning: Mismatch in timestep count. Adjusting to match existing data.")
            qdub_series = qdub_series.reindex(df.index, fill_value=0)  # fill missing values with 0

        # append new simulation as a new column
        df[simulation_name] = qdub_series

    else:
        # if the file does not exist, create a new dataframe
        df = pd.DataFrame({simulation_name: qdub_series})

        # add a timestep index column
        df.insert(0, "Timestep", range(1, len(qdub) + 1))  # ensures the first column is the timestep

    # save the updated dataframe to csv
    #df.to_csv(output_csv)
    df.to_csv(csv_outpath, mode='w', index=True)

    print(f"heat flux data for simulation '{simulation_name}' saved to {output_csv}.")

# ---------------------- start program ------------------------ 

offset = 0
point_data = {}
if(not('fields_to_write' in locals())):
  print('not in locals')
  fields_to_write = []
if(not('fields_to_write' in globals())):
  print('not in globals')
#print('fields_to_write' in globals())

if not os.path.exists(outdir):  
    os.makedirs(outdir)

gas_field_names,fuel_field_names,div_by_dens,fields_to_write = formOutputList(gridlist_pf,fields_to_write,nfuel)
fname   = indir + readfilename
XI, YI, ZI, volume = metrics(topofile, Nx, Ny, Nz, dx, dy, dz, aa1, f0, stretch)
# point_data.update(read_fields(fname, Nx, Ny, Nz, Nzfuel, initial, gas_field_names, fuel_field_names, div_by_dens)) 

# compute initial fuel density before entering loop
# rho_fuel_tot_initial, rho_fuel_tot_final, initial_fuel_density, consumption = total_fuel_consumption(
#     "./comp.out.1000",
#     "./comp.out.120000",
#     Nx, Ny, Nz, Nzfuel,
#     gas_field_names,
#     fuel_field_names,
#     div_by_dens
# ) 

rho_fuel_initial1, rho_fuel_initial2, rho_fuel_tot_initial, rho_fuel_tot_final, initial_fuel_density, consumption = total_fuel_consumption(
    "./comp.out.1000",
    "./comp.out.120000",
    Nx, Ny, Nz, Nzfuel,
    gas_field_names,
    fuel_field_names,
    div_by_dens
)

# MC * Cp_h2o + % solid * Cp_grass. has to be hardcoded every time bc info lives in fuellist, not GL
#-----------------study 1-----------------------
cp_solid1 = (0.25 * 4184 + 0.75 * 3000) #allc1, wet
cp_solid2 = (0.25 * 4184 + 0.75 * 3000) #allc1, dry

# cp_solid1 = (0.7 * 4184 + 0.3 * 3000) #s1c2
# cp_solid2 = (0.2 * 4184 + 0.8 * 3000) #s1c2

# cp_solid1 = (1.15 * 4184 + 0.1 * 3000) #s1c3, wet 
# cp_solid2 = (0.15 * 4184 + 0.85 * 3000) #s1c3, dry

# cp_solid1 = (1.6 * 4184 + 0.1 * 3000) #s1c4, wet
# cp_solid2 = (0.1 * 4184 + 0.9 * 3000) #s1c4, dry

# cp_solid1 = (2.5 * 4184 + 0.1 * 3000) #s1c5, wet
# cp_solid2 = (0 * 4184 + 1 * 3000) #s1c5, dry 

#-----------------study 2-----------------------
#1 above

# cp_solid1 = (0.396 * 4184 + (1-0.396) * 3000) #s2c2
# cp_solid2 = (0.187 * 4184 + (1-0.187) * 3000) #s2c2

# cp_solid1 = (0.542 * 4184 + (1-0.542) * 3000) #s2c3, wet
# cp_solid2 = (0.125 * 4184 + (1-0.125) * 3000) #s2c3, dry

# cp_solid1 = (0.688 * 4184 + (1-0.688) * 3000) #s2c4, wet
# cp_solid2 = (0.062 * 4184 + (1-0.062) * 3000) #s2c4, dry

# cp_solid1 = (0.834 * 4184 + (1-0.834) * 3000) #s2c5, wet
# cp_solid2 = (0. * 4184 + (1) * 3000) #s2c5, dry

#-----------------study 3-----------------------
#1 above

# cp_solid1 = (0.3125 * 4184 + (1-0.3125) * 3000) #s3c2
# cp_solid2 = (0.188 * 4184 + (1-0.188) * 3000) #s3c2

# cp_solid1 = (0.375 * 4184 + (1-0.375) * 3000) #s3c3, wet
# cp_solid2 = (0.125 * 4184 + (1-0.125) * 3000) #s3c3, dry

# cp_solid1 = (0.4375 * 4184 + (1-0.4375) * 3000) #s3c4, wet
# cp_solid2 = (0.0625 * 4184 + (1-0.0625) * 3000) #s3c4, dry

# cp_solid1 = (0.5 * 4184 + 0.5 * 3000) #s3c5, wet
# cp_solid2 = (0 * 4184 + 1 * 3000) #s3c5, dry
#--------------------ded-------------------------

# cp_solid1 = (0.1 * 4184 + .9 * 3000) #allc1, dry
# cp_solid2 = cp_solid1

#--------------------liv-------------------------

# cp_solid1 = (0.9 * 4184 + 0.1 * 3000) #allc1, wet
# cp_solid2 = cp_solid1

#------------------------------------------------

u_dict = []
v_dict = []
w_dict = []
theta_dict = [] 
qdub = []

if nfuel==1:
    ftemp = []

if nfuel==2:
    ftemp1 = []
    ftemp2 = []

fire_front_positions = []
spread_rates = []
previous_position_x = None # change from logical to ==0?
previous_time = None
flame_maps = []
flame_counts = []
flame_depths = []
        
# for i in range(initial, final, incr): 
#     filename = fname+str(i)
#     vtsfile = outdir+outname+str(i)
#     if not os.path.exists(filename):
#         continue  # skip missing files
    
#     f = open(filename, 'rb')
#     point_data.update(read_fields(fname, Nx, Ny, Nz, Nzfuel, i, gas_field_names, fuel_field_names, div_by_dens))
#     timestep_qdub = heat_flux(point_data, Nx, Ny, cp_solid1, cp_solid2, nfuel) #  ftemp1,ftemp2
#     if nfuel == 1: 
#         ftemp += ftemp
#     if nfuel == 2:
#         ftemp1 += ftemp1 
#         ftemp2 += ftemp2 

for i in range(initial, final, incr): 
    filename = fname+str(i)
    vtsfile = outdir+outname+str(i)
    if not os.path.exists(filename):
        continue  # skip missing files
    
    f = open(filename, 'rb')
    point_data.update(read_fields(fname, Nx, Ny, Nz, Nzfuel, i, gas_field_names, fuel_field_names, div_by_dens))
    
    # compute heat flux with reaction heat
    timestep_qdub, ftemp_1, ftemp_2 = heat_flux(
        point_data, Nx, Ny, cp_solid1, cp_solid2, nfuel, 
        rho_fuel_initial1, rho_fuel_initial2
    )
    qdub.append(timestep_qdub)  # store heat flux value
    
    if nfuel == 1 and ftemp_1 is not None: 
        ftemp.append(np.max(ftemp_1))  # store max temp
    if nfuel == 2:
        if ftemp_1 is not None:
            ftemp1.append(np.max(ftemp_1))
        if ftemp_2 is not None:
            ftemp2.append(np.max(ftemp_2))

    #if os.path.isfile(vtsfile+'.vts'): # remove old vts' 
     #   os.remove(vtsfile+'.vts')
        
    u_dict.append(point_data["u"].copy()) 
    v_dict.append(point_data["v"].copy()) 
    w_dict.append(point_data["w"].copy()) 
    theta_dict.append(point_data["theta"].copy()) 

    # fire spread 
    fire_front_x, spread_rate = fire_spread(
        point_data, Nx, Ny, dx, initial_fuel_density, i*0.001, previous_position_x, previous_time
    )
    
    if fire_front_x is not None: 
        fire_front_positions.append((i, fire_front_x))

        if spread_rate is not None: 
            spread_rates.append((i, spread_rate))

        # update previous values
        previous_position_x = fire_front_x
        previous_time = i*0.001
        
        # flame depth/detection/map
        # print(f"timestep {i}: fire front at {fire_front_x} m")  # debug print
        
        flame_map, flame_count, max_flame_depth = flame_depth(point_data, Nx, Ny, fire_front_x, dx)
        # print(f"timestep {i}: flame count: {flame_counts}, depth: {flame_depths} m")  # debug print
        
        flame_maps.append(flame_map)
        flame_counts.append((i, flame_count))  # save with timestep info 
        flame_depths.append((i, max_flame_depth)) # save maximum flame depth for this timestep
        
    # theta plot at final
    # theta_plt = plt.imshow(point_data['theta'][:,:,0].T, origin='lower', cmap='RdYlBu') #, vmin=vmin, vmax=vmax)
    # plt.title(f'Potential Temperature (gas) @ {i * 0.01:.1f} seconds')
    # plt.colorbar(theta_plt)
    # plt.show()
    # #plt.savefig(os.path.join(outdir, f'theta @ {i} steps.png'))
    # plt.clf()
    
    # if filename.endswith("5000"):
    #     plt.xlim(0, Nx)
    #     plt.ylim(0, Ny)
        
    #     # u plot at final
    #     u_vel_plt = plt.imshow(point_data['u'][:,:,1].T, origin='lower', cmap='RdYlBu') #, vmin=vmin, vmax=vmax)
    #     plt.title(f'U velocity @ {i * 0.01:.1f} seconds')
    #     plt.colorbar(u_vel_plt)
    #     #plt.show()
    #     #plt.savefig(os.path.join(outdir, f'U @ {i} steps.png'))
    #     plt.clf()
        
    #     # theta plot at final
    #     theta_plt = plt.imshow(point_data['theta'][:,:,1].T, origin='lower', cmap='RdYlBu') #, vmin=vmin, vmax=vmax)
    #     plt.title(f'Potential Temperature (gas) @ {i * 0.01:.1f} seconds')
    #     plt.colorbar(theta_plt)
    #     #plt.show()
    #     #plt.savefig(os.path.join(outdir, f'theta @ {i} steps.png'))
    #     plt.clf()
        
    #     # fuel density 1 at final 
    #     fuel_density_plt = plt.imshow(point_data['rhoFuel_1'][:,:,0].T, origin='lower', cmap='RdYlBu') #, vmin=vmin, vmax=vmax)
    #     plt.title(f'fuel density 1 @ {i * 0.01:.1f} seconds')
    #     plt.colorbar(fuel_density_plt)
    #     #plt.show()
    #     #plt.savefig(os.path.join(outdir, f'fuel density @ {i} steps.png'))
    #     plt.clf()
        
    #     # fuel density 2 at final 
    #     fuel_density_plt2 = plt.imshow(point_data['rhoFuel_2'][:,:,0].T, origin='lower', cmap='RdYlBu') #, vmin=vmin, vmax=vmax)
    #     plt.title(f'fuel density 2 @ {i * 0.01:.1f} seconds')
    #     plt.colorbar(fuel_density_plt2)
    #     #plt.show()
    #     #plt.savefig(os.path.join(outdir, f'fuel density 2 @ {i} steps.png'))
    #     plt.clf()
    
    # call function to make VTKs
    #gridToVTK(vtsfile, XI, YI, ZI, pointData = select_data(point_data, fields_to_write)) # ,fuel_field_names)) 

u_array = np.array(u_dict)
v_array = np.array(v_dict)
w_array = np.array(w_dict)
theta_array = np.array(theta_dict)

# calculate min/max across all stored timesteps
u_max = np.max(u_array)
u_min = np.min(u_array)
#print(f'u min: {u_min:.3f} u max: {u_max:.3f}') 

v_max = np.max(v_array)
v_min = np.min(v_array)
#print(f'v min: {v_min:.3f} v max: {v_max:.3f}') 

w_max = np.max(w_array)
w_min = np.min(w_array)
#print(f'w min: {w_min:.3f} w max: {w_max:.3f}') 

theta_max = np.max(theta_array)
theta_min = np.min(theta_array)
#print(f'theta min: {theta_min:.3f} theta max: {theta_max:.3f}') 

# if nfuel==1:
#     print(f'max fuel temp: {np.max(ftemp):.3f}')

# # print(f'ftemp1: ', ftemp1) 
# if nfuel==2: 
#     print(f'fuel_temp1_array max: {np.max(ftemp1):.3f}') 
#     print(f'fuel_temp2_array max: {np.max(ftemp2):.3f}') 

# compute overall fire spread rate
overall_spread_rate = np.mean([rate for _, rate in spread_rates]) if spread_rates else 0
print(f"overall fire spread rate: {overall_spread_rate:.3f} m/s")
print(f"fuel consumption (f/i): {consumption:.3f}") 

# flame depth etc
# print(f"flame count: {flame_counts} cells.")
# print(f"max local depths (step, local max): {flame_depths}") 

# max_depth_timestep = np.argmax(flame_depths[:, 1])
# max_flame_depth_overall = flame_depths[max_depth_timestep, 1]

# print(f"maximum overall flame depth: {max_flame_depth_overall:.2f} meters")
# print(f"at timestep {max_depth_timestep}")

max_flame_depth_overall = max(depth for _, depth in flame_depths)
max_depth_timestep = next(i for i, (_, depth) in enumerate(flame_depths) if depth == max_flame_depth_overall)

print(f"maximum flame depth was {max_flame_depth_overall:.2f} meters")
#print(f"occurred at timestep {max_depth_timestep}")

# plot q" 
#print('timestep_qdub : ', timestep_qdub)
#qdub_plt = plt.plot(qdub) #,origin='lower')
#plt.show() 

# store qdubs across all sims
simulation_name = "0s1c1"  # Unique identifier for this run
# store_csv(qdub, simulation_name, csv_outpath) #output_csv="heat_flux_results.csv")