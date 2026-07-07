"""
Postprocess FIRETEC comp.out.* binary output for one run into CSVs:
fuel consumption, energy release (total/convective/radiative), spread rate,
flame depth, combustion efficiency, ignition success, fire-atmosphere
feedback (turbulence/wind), fuel preheat/drying, and plume dynamics.

nfuel and the fuel specific heats (cp_solid1/cp_solid2) are read
automatically from the run's own gridlist/fuellist -- no per-run hand-editing
needed. Run from anywhere; point --indir at the run directory (defaults to
the current directory).

usage:
    python firetec_binary_postprocess.py --simulation-name 0s1c1
    python firetec_binary_postprocess.py --simulation-name 501 --write-vtk
"""
import argparse
import os

import numpy as np

from postprocess.io_fields import formOutputList, read_fields, select_data, find_available_timesteps
from postprocess.grid import metrics
from postprocess.fuel_state import fuel_consumption
from postprocess.combustion import consumption_and_reaction_heat
from postprocess.fire_spread import fire_spread
from postprocess.heat_flux import heat_flux
from postprocess.flame import flame_depth
from postprocess.turbulence import turbulence_wind_coupling
from postprocess.preheat import fuel_preheat_drying
from postprocess.plume import plume_dynamics
from postprocess.storage import store_all_data
from postprocess.fuel_properties import cp_solid_from_fuellist


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--simulation-name', default=None,
                   help='unique identifier for this run (default: current directory name)')
    p.add_argument('--indir', default=os.getcwd(), help='directory containing comp.out.* files')
    p.add_argument('--gridlist-pf', default=None, help='directory containing gridlist (default: --indir)')
    p.add_argument('--fuellist', default=None, help='path to fuellist (default: <indir>/fuellist)')
    p.add_argument('--outdir', default='./postprocessing/', help='VTK output directory')
    p.add_argument('--csv-root-dir', default='/Users/michellegee/LANL/het_fuels/csvs/')
    p.add_argument('--nfuel', type=int, default=None,
                   help='override auto-detected nfuel (normally read from gridlist)')
    p.add_argument('--initial', type=int, default=4000, help='timestep of ignition')
    p.add_argument('--final', type=int, default=65001, help='timestep to stop at')
    p.add_argument('--incr', type=int, default=1000)
    p.add_argument('--write-vtk', action='store_true', help='write a VTK file every timestep (slow, large)')

    # domain -- constant across this run ensemble; override if needed
    p.add_argument('--nx', type=int, default=200)
    p.add_argument('--ny', type=int, default=100)
    p.add_argument('--nz', type=int, default=41)
    p.add_argument('--nzfuel', type=int, default=1)
    p.add_argument('--dx', type=float, default=2.0)
    p.add_argument('--dy', type=float, default=2.0)
    p.add_argument('--dz', type=float, default=15.0)
    p.add_argument('--aa1', type=float, default=0.1)
    p.add_argument('--f0', type=float, default=0.0)
    p.add_argument('--stretch', type=int, default=2, help='0=none, 1=tanh (unimplemented), 2=cubic')
    p.add_argument('--topofile', default='', help='leave empty for flat terrain')

    # fuel specific heat -- auto-computed from the fuellist's gmoisture unless overridden
    p.add_argument('--cp-solid1', type=float, default=None)
    p.add_argument('--cp-solid2', type=float, default=None)

    # overall consumption ratio is computed between these two snapshots; auto-detected
    # from whatever comp.out.* files are actually present (some runs stop early and
    # never reach a later snapshot another run's "final" happens to be hardcoded to)
    p.add_argument('--consumption-initial-step', type=int, default=None)
    p.add_argument('--consumption-final-step', type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    indir = args.indir
    gridlist_pf = args.gridlist_pf or indir
    fuellist_path = args.fuellist or os.path.join(indir, 'fuellist')
    simulation_name = args.simulation_name or os.path.basename(os.path.normpath(indir))
    readfilename = '/comp.out.'
    fname = indir + readfilename

    if args.write_vtk and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    gas_field_names, fuel_field_names, div_by_dens, fields_to_write, detected_nfuel = formOutputList(
        gridlist_pf, [], args.nfuel or 1)
    nfuel = args.nfuel or detected_nfuel
    print(f"simulation '{simulation_name}': nfuel={nfuel} (from {'--nfuel override' if args.nfuel else 'gridlist'})")

    XI, YI, ZI, volume = metrics(args.topofile, args.nx, args.ny, args.nz, args.dx, args.dy, args.dz,
                                 args.aa1, args.f0, args.stretch)

    # overall consumption ratio: use the first/last comp.out.* snapshots actually
    # present, unless explicitly overridden -- don't assume every run reached the
    # same final timestep
    available_steps = find_available_timesteps(indir, readfilename)
    if not available_steps:
        raise FileNotFoundError(f"no comp.out.* files found in {indir}")
    consumption_initial_step = args.consumption_initial_step or available_steps[0]
    consumption_final_step = args.consumption_final_step or available_steps[-1]
    comp_out_initial = f"./comp.out.{consumption_initial_step}"
    comp_out_final = f"./comp.out.{consumption_final_step}"

    # compute initial fuel density before entering loop
    if nfuel == 1:
        rho_fuel_initial, rho_fuel_tot_initial, rho_fuel_tot_final, initial_fuel_density, consumption, rho_water_initial = fuel_consumption(
            comp_out_initial, comp_out_final, args.nx, args.ny, args.nz, args.nzfuel, nfuel,
            gas_field_names, fuel_field_names, div_by_dens)
        rho_fuel_initial1 = rho_fuel_initial2 = None
    else:
        (rho_fuel_initial1, rho_fuel_initial2, rho_fuel_tot_initial, rho_fuel_tot_final, initial_fuel_density,
         consumption, rho_water_initial1, rho_water_initial2) = fuel_consumption(
            comp_out_initial, comp_out_final, args.nx, args.ny, args.nz, args.nzfuel, nfuel,
            gas_field_names, fuel_field_names, div_by_dens)
        rho_water_initial = None

    # fuel specific heat -- auto-computed from this run's own fuellist moisture content
    # (cp = MC*Cp_water + Cp_dry) / (1+MC); see postprocess/fuel_properties.py for why the
    # naive MC*Cp_water + (1-MC)*Cp_dry blend is wrong once live-fuel MC exceeds 100%)
    auto_cp_solid1, auto_cp_solid2 = cp_solid_from_fuellist(fuellist_path, nfuel)
    cp_solid1 = args.cp_solid1 if args.cp_solid1 is not None else auto_cp_solid1
    cp_solid2 = args.cp_solid2 if args.cp_solid2 is not None else (auto_cp_solid2 if nfuel == 2 else None)
    print(f"cp_solid1={cp_solid1:.1f} J/kg-K" + (f", cp_solid2={cp_solid2:.1f} J/kg-K" if nfuel == 2 else ""))

    # initialize arrays for storage
    qdub = []
    q_conv_list = []
    q_rad_list = []
    if nfuel == 1:
        avg_consumption = []
        consumption_max_rate = []
        combustion_efficiency = []
    if nfuel == 2:
        avg_f1_consumption = []
        avg_f2_consumption = []
        avg_tot_consumption = []
        consumption_max_rate_f1 = []
        consumption_max_rate_f2 = []
        combustion_efficiency_f1 = []
        combustion_efficiency_f2 = []
    consumption_max_rate_total = []

    fire_front_positions = []
    spread_rates = []
    previous_position_x = None
    previous_time = None
    flame_depths = []

    tke_near_front_list = []
    tke_ambient_list = []
    wind_near_front_list = []
    wind_ambient_list = []
    drying_fraction_list = []
    preheat_zone_width_list = []
    w_max_list = []
    plume_rise_height_list = []

    point_data = {}
    for i in range(args.initial, args.final, args.incr):
        filename = fname + str(i)
        if not os.path.exists(filename):
            continue  # skip missing files

        point_data.update(read_fields(fname, args.nx, args.ny, args.nz, args.nzfuel, i,
                                      gas_field_names, fuel_field_names, div_by_dens))

        # compute heat flux with reaction heat
        if nfuel == 1:
            timestep_qdub, timestep_q_conv, timestep_q_rad, ftemp, _ = heat_flux(
                point_data, cp_solid1, None, nfuel, rho_fuel_initial, None)
            qdub.append(timestep_qdub)
            q_conv_list.append(timestep_q_conv)
            q_rad_list.append(timestep_q_rad)

            frhof, reactFuelGas, timestep_combustion_efficiency = consumption_and_reaction_heat(
                nfuel, point_data['density'][:, :, 0],
                point_data.get('kb', np.zeros((args.nx, args.ny, args.nz)))[:, :, 0],
                point_data.get('O2', 0.23 * np.ones((args.nx, args.ny, args.nz)))[:, :, 0],
                rho_fuel_initial, None,
                point_data['rhoFuel'][:, :, 0] if 'rhoFuel' in point_data else np.zeros((args.nx, args.ny)), None,
                point_data['rho_water'][:, :, 0] if 'rho_water' in point_data else np.zeros((args.nx, args.ny)), None,
                ftemp if ftemp is not None else np.zeros((args.nx, args.ny)), None)
            combustion_efficiency.append(timestep_combustion_efficiency)

            max_fc = np.max(frhof) if frhof is not None else 0
            avg_consumption.append(np.mean(frhof) if frhof is not None else 0)
            consumption_max_rate.append(max_fc)
            consumption_max_rate_total.append(max_fc)

        if nfuel == 2:
            timestep_qdub, timestep_q_conv, timestep_q_rad, ftemp_1, ftemp_2 = heat_flux(
                point_data, cp_solid1, cp_solid2, nfuel, rho_fuel_initial1, rho_fuel_initial2)
            qdub.append(timestep_qdub)
            q_conv_list.append(timestep_q_conv)
            q_rad_list.append(timestep_q_rad)

            frhof1, frhof2, reactFuelGas1, reactFuelGas2, timestep_ce1, timestep_ce2 = consumption_and_reaction_heat(
                nfuel, point_data['density'][:, :, 0],
                point_data.get('kb', np.zeros((args.nx, args.ny, args.nz)))[:, :, 0],
                point_data.get('O2', 0.23 * np.ones((args.nx, args.ny, args.nz)))[:, :, 0],
                rho_fuel_initial1, rho_fuel_initial2,
                point_data['rhoFuel_1'][:, :, 0] if 'rhoFuel_1' in point_data else np.zeros((args.nx, args.ny)),
                point_data['rhoFuel_2'][:, :, 0] if 'rhoFuel_2' in point_data else np.zeros((args.nx, args.ny)),
                point_data['rho_water1'][:, :, 0] if 'rho_water1' in point_data else np.zeros((args.nx, args.ny)),
                point_data['rho_water2'][:, :, 0] if 'rho_water2' in point_data else np.zeros((args.nx, args.ny)),
                ftemp_1 if ftemp_1 is not None else np.zeros((args.nx, args.ny)),
                ftemp_2 if ftemp_2 is not None else np.zeros((args.nx, args.ny)))
            combustion_efficiency_f1.append(timestep_ce1)
            combustion_efficiency_f2.append(timestep_ce2)

            max_fc_f1 = np.max(frhof1) if frhof1 is not None else 0
            max_fc_f2 = np.max(frhof2) if frhof2 is not None else 0
            avg_f1_consumption.append(np.mean(frhof1) if frhof1 is not None else 0)
            avg_f2_consumption.append(np.mean(frhof2) if frhof2 is not None else 0)
            avg_tot_consumption.append(np.mean(frhof1 + frhof2) if (frhof1 is not None and frhof2 is not None) else 0)
            consumption_max_rate_f1.append(max_fc_f1)
            consumption_max_rate_f2.append(max_fc_f2)
            consumption_max_rate_total.append(np.max(frhof1 + frhof2) if (frhof1 is not None and frhof2 is not None) else 0)

        # fire spread
        fire_front_x, spread_rate = fire_spread(
            nfuel, point_data, args.nx, args.dx, initial_fuel_density, i * 0.01, previous_position_x, previous_time)

        fire_front_positions.append((i, fire_front_x))
        spread_rates.append((i, spread_rate))
        previous_position_x = fire_front_x
        previous_time = i * 0.01

        # flame depth/detection/map
        flame_map, flame_count, max_flame_depth = flame_depth(point_data, args.nx, args.ny, fire_front_x, args.dx)
        flame_depths.append((i, max_flame_depth))

        # fire-atmosphere feedback: turbulence & wind, near-front vs. ambient
        fire_front_index = int(fire_front_x / args.dx)
        tke_near, tke_ambient, wind_near, wind_ambient = turbulence_wind_coupling(
            point_data, fire_front_index, args.nx, args.dx)
        tke_near_front_list.append(tke_near)
        tke_ambient_list.append(tke_ambient)
        wind_near_front_list.append(wind_near)
        wind_ambient_list.append(wind_ambient)

        # fuel preheat/drying ahead of the front
        if nfuel == 1:
            drying_fraction, preheat_zone_width = fuel_preheat_drying(
                point_data, fire_front_index, args.nx, args.dx, nfuel, rho_water_initial)
        else:
            drying_fraction, preheat_zone_width = fuel_preheat_drying(
                point_data, fire_front_index, args.nx, args.dx, nfuel, rho_water_initial1, rho_water_initial2)
        drying_fraction_list.append(drying_fraction)
        preheat_zone_width_list.append(preheat_zone_width)

        # plume dynamics above the flame footprint
        w_max, plume_rise_height = plume_dynamics(point_data, flame_map, ZI)
        w_max_list.append(w_max)
        plume_rise_height_list.append(plume_rise_height)

        if args.write_vtk:
            from pyevtk.hl import gridToVTK
            vtsfile = args.outdir + 'vtk_output.' + str(i)
            gridToVTK(vtsfile, XI, YI, ZI, pointData=select_data(point_data, fields_to_write))

    overall_spread_rate = np.mean([rate for _, rate in spread_rates]) if spread_rates else 0
    print(f"overall fire spread rate: {overall_spread_rate:.3f} m/s")
    print(f"fuel consumption (f/i): {consumption:.3f}")

    spread_rate_values = np.array([rate for _, rate in spread_rates])
    flame_depth_values = np.array([depth for _, depth in flame_depths])

    # ignition success: did the fire advance well past the ignition line, and was it
    # still spreading (not stalled/self-extinguished) in the back half of the run?
    # both are run-level scalars, broadcast to a constant time series below so they
    # fit the same per-timestep CSV storage as everything else.
    front_positions_only = [pos for _, pos in fire_front_positions]
    total_front_advance = (front_positions_only[-1] - front_positions_only[0]) if len(front_positions_only) > 1 else 0.0
    half = len(spread_rate_values) // 2
    late_window_mean_spread_rate = float(np.mean(spread_rate_values[half:])) if len(spread_rate_values) > half else 0.0
    ignition_success = bool(total_front_advance > 10 * args.dx and late_window_mean_spread_rate > 0)
    print(f"ignition success: {ignition_success} (total front advance: {total_front_advance:.1f} m, "
          f"late-window mean spread rate: {late_window_mean_spread_rate:.3f} m/s)")

    grid_area = args.nx * args.ny * args.dx * args.dy
    qdub = np.array(qdub)
    q_conv_array = np.array(q_conv_list)
    q_rad_array = np.array(q_rad_list)
    total_energy_release_rate = np.divide(qdub, grid_area)

    common_metrics = {
        'qdub': qdub,
        'total_energy_release_rate': total_energy_release_rate,
        'q_conv': q_conv_array,
        'q_rad': q_rad_array,
        'spread_rates': spread_rate_values,
        'flame_depth': flame_depth_values,
        'tke_near_front': np.array(tke_near_front_list),
        'tke_ambient': np.array(tke_ambient_list),
        'wind_near_front': np.array(wind_near_front_list),
        'wind_ambient': np.array(wind_ambient_list),
        'drying_fraction': np.array(drying_fraction_list),
        'preheat_zone_width': np.array(preheat_zone_width_list),
        'w_max': np.array(w_max_list),
        'plume_rise_height': np.array(plume_rise_height_list),
        'ignition_success': np.full(len(spread_rate_values), ignition_success),
        'total_front_advance': np.full(len(spread_rate_values), total_front_advance),
    }

    if nfuel == 1:
        store_all_data(args.csv_root_dir, args.initial, args.final, args.incr, simulation_name, {
            **common_metrics,
            'consumption_max_rate': np.array(consumption_max_rate),
            'consumption_max_rate_total': np.array(consumption_max_rate_total),
            'avg_consumption': np.array(avg_consumption),
            'avg_tot_consumption': np.array(avg_consumption),
            'fc_normalized': np.divide(consumption_max_rate, consumption_max_rate_total,
                                       where=np.array(consumption_max_rate_total) != 0,
                                       out=np.zeros(len(consumption_max_rate))),
            'combustion_efficiency': np.array(combustion_efficiency),
        })

    if nfuel == 2:
        cmr_f1 = np.array(consumption_max_rate_f1)
        cmr_f2 = np.array(consumption_max_rate_f2)
        cmr_total = np.array(consumption_max_rate_total)
        store_all_data(args.csv_root_dir, args.initial, args.final, args.incr, simulation_name, {
            **common_metrics,
            'consumption_max_rate_f1': cmr_f1,
            'consumption_max_rate_f2': cmr_f2,
            'consumption_max_rate_total': cmr_total,
            'avg_f1_consumption': np.array(avg_f1_consumption),
            'avg_f2_consumption': np.array(avg_f2_consumption),
            'avg_tot_consumption': np.array(avg_tot_consumption),
            'f1fc_normalized': np.divide(cmr_f1, cmr_total, where=cmr_total != 0, out=np.zeros(len(cmr_f1))),
            'f2fc_normalized': np.divide(cmr_f2, cmr_total, where=cmr_total != 0, out=np.zeros(len(cmr_f2))),
            'combustion_efficiency_f1': np.array(combustion_efficiency_f1),
            'combustion_efficiency_f2': np.array(combustion_efficiency_f2),
        })


if __name__ == '__main__':
    main()
