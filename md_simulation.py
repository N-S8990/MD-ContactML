import argparse
import logging
import os
import sys
import shutil
import numpy as np

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer
except ImportError:
    print("Error: OpenMM or PDBFixer is not installed. Please install them via conda or pip.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_INPUT_PDB = "1A22.pdb"


def prepare_system(input_pdb, output_pdb):

    logger.info(f"Preparing system from {input_pdb}...")
    fixer = PDBFixer(filename=input_pdb)
    
    logger.info("Finding missing residues (and ignoring them to prevent massive box sizes)...")
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    
    logger.info("Finding nonstandard residues...")
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    logger.info("Removing heterogens (including crystallographic water) for Implicit Solvent...")
    fixer.removeHeterogens(False)
    
    logger.info("Adding missing sidechain atoms...")
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    logger.info("Adding missing hydrogens...")
    fixer.addMissingHydrogens(7.0)

    logger.info("Centering molecule at (0,0,0)...")
    positions = np.array(fixer.positions.value_in_unit(unit.nanometers))
    center = np.mean(positions, axis=0)
    positions -= center
    fixer.positions = [mm.Vec3(*pos) for pos in positions] * unit.nanometers
    
    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
    logger.info(f"Prepared system saved to {output_pdb}")
    return fixer.topology, fixer.positions

def run_simulation(topology, positions, out_dir, prefix, steps=10000, platform_name=None,
                   max_min_iters=0, seed=None, report_interval=50000, temperature=300):
    logger.info("Setting up simulation parameters...")
    forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    
    platform = None
    if platform_name:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            logger.info(f"Using platform: {platform_name}")
        except Exception as e:
            logger.warning(f"Could not load platform {platform_name}: {e}. Using default.")
            
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=max_min_iters)

    if seed is None:
        simulation.context.setVelocitiesToTemperature(10 * unit.kelvin)
    else:
        simulation.context.setVelocitiesToTemperature(10 * unit.kelvin, seed)
        
    logger.info("Warming up the system gently...")
    integrator.setStepSize(0.0001 * unit.picoseconds)
    simulation.step(1000)
    
    integrator.setTemperature(temperature * unit.kelvin)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    integrator.setStepSize(0.001 * unit.picoseconds)
    
    os.makedirs(out_dir, exist_ok=True)
    dcd_path = os.path.join(out_dir, f"{prefix}_traj.dcd")
    log_path = os.path.join(out_dir, f"{prefix}_sim.log")
    
    simulation.reporters.append(app.DCDReporter(dcd_path, max(1, report_interval)))
    simulation.reporters.append(app.StateDataReporter(log_path, max(1, report_interval), step=True, 
                                                      potentialEnergy=True, temperature=True))
                                                      
    print_interval = max(1, steps // 100)
    simulation.reporters.append(app.StateDataReporter(sys.stdout, print_interval, step=True, 
                                                      potentialEnergy=True, temperature=True, 
                                                      progress=True, remainingTime=True, 
                                                      speed=True, totalSteps=steps, separator='\t'))
                                                      
    logger.info(f"Running simulation for {steps} steps at {temperature}K...")
    simulation.step(steps)
    logger.info(f"Simulation complete. Trajectory saved to {dcd_path}")


def run_temperature_series(input_pdb, replica_dir, prefix, args, seed=None):
    temperatures = getattr(args, 'temperatures', [300, 400])
    for temp in temperatures:
        temp_dir = os.path.join(replica_dir, f"temp_{temp}K")
        dcd_path = os.path.join(temp_dir, f"{prefix}_{temp}K_traj.dcd")

        if os.path.exists(dcd_path) and getattr(args, 'skip_existing', False):
            logger.info(f"Trajectory already exists at {dcd_path}. Skipping due to --skip_existing.")
            continue

        if os.path.exists(temp_dir):
            logger.info(f"Removing older data in {temp_dir}...")
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        prep_pdb = os.path.join(temp_dir, f"{prefix}_{temp}K_prepared.pdb")
        topology, positions = prepare_system(input_pdb, prep_pdb)
        run_simulation(
            topology,
            positions,
            temp_dir,
            f"{prefix}_{temp}K",
            args.steps,
            args.platform,
            args.max_min_iters,
            seed,
            args.report_interval,
            temp,
        )


def get_replica_dir(base_out_dir, rep_id):
    norm_path = os.path.normpath(base_out_dir)
    basename = os.path.basename(norm_path)
    if basename == f"replica_{rep_id}":
        return norm_path
    return os.path.join(base_out_dir, f"replica_{rep_id}")


def run_replicas(input_pdb, base_out_dir, prefix, args):
    if args.replica_id is not None:
        replicas_to_run = [args.replica_id]
    else:
        norm_path = os.path.normpath(base_out_dir)
        basename = os.path.basename(norm_path)
        if basename.startswith("replica_"):
            try:
                rep_num = int(basename.split("_")[1])
                replicas_to_run = [rep_num]
            except ValueError:
                replicas_to_run = list(range(1, args.num_replicas + 1))
        else:
            replicas_to_run = list(range(1, args.num_replicas + 1))

    for rep_id in replicas_to_run:
        replica_dir = get_replica_dir(base_out_dir, rep_id)
        logger.info(f"============================================================")
        logger.info(f"STARTING SIMULATION FOR REPLICA {rep_id} IN: {replica_dir}")
        logger.info(f"============================================================")
        rep_seed = (args.seed + (rep_id - 1) * 1000) if args.seed is not None else (1000 * rep_id + 42)
        run_temperature_series(input_pdb, replica_dir, prefix, args, seed=rep_seed)


def main():
    parser = argparse.ArgumentParser(description="Run MD Simulation with OpenMM across Replicas and Temperatures")
    parser.add_argument("--input_pdb", type=str, default=DEFAULT_INPUT_PDB, help="Input PDB file (default: 1A22.pdb)")
    parser.add_argument("--out_dir", type=str, default="data", help="Base output directory (default: data)")
    parser.add_argument("--num_replicas", type=int, default=3, help="Number of independent replicas to simulate (default: 3)")
    parser.add_argument("--replica_id", type=int, default=None,
                        help="Run only a specific replica ID (e.g. 1, 2, or 3). Default: None (runs all num_replicas).")
    parser.add_argument("--temperatures", nargs="+", type=int, default=[300, 400],
                        help="Temperatures in Kelvin to simulate (default: 300 400)")
    parser.add_argument("--wild_type_pdb", type=str, default=None,
                        help="Optional wild-type PDB for additional m_w data")
    parser.add_argument("--mutant_pdb", type=str, default=None,
                        help="Optional mutant PDB for additional m_w data")
    parser.add_argument("--mw_out_dir", type=str, default="data/m_w",
                        help="Output directory for wild-type and mutant data")
    parser.add_argument("--prefix", type=str, default="sim", help="Prefix for output files (default: sim)")
    parser.add_argument("--steps", type=int, default=200000,
                        help="Number of MD steps to run (default: 200000; 200 ps at 1 fs, yielding 200 frames)")
    parser.add_argument("--platform", type=str, choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], 
                        default='CUDA', help="Compute platform to use (default: CUDA)")
    parser.add_argument("--max_min_iters", type=int, default=0, help="Max iterations for energy minimization (0 for unlimited)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for velocity initialization (automatically offset per replica).")
    parser.add_argument("--report_interval", type=int, default=1000,
                        help="Save one DCD frame every N steps (default: 1000).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip simulation if trajectory file (.dcd) already exists in target directory.")
    
    args = parser.parse_args()

    if (args.wild_type_pdb is None) != (args.mutant_pdb is None):
        parser.error("--wild_type_pdb and --mutant_pdb must be provided together.")
    if args.wild_type_pdb and not os.path.isfile(args.wild_type_pdb):
        parser.error(f"Wild-type PDB not found: {args.wild_type_pdb}")
    if args.mutant_pdb and not os.path.isfile(args.mutant_pdb):
        parser.error(f"Mutant PDB not found: {args.mutant_pdb}")

    run_replicas(args.input_pdb, args.out_dir, args.prefix, args)

    if args.wild_type_pdb and args.mutant_pdb:
        run_replicas(
            args.wild_type_pdb,
            os.path.join(args.mw_out_dir, "wild_type"),
            f"{args.prefix}_wt",
            args,
        )
        run_replicas(
            args.mutant_pdb,
            os.path.join(args.mw_out_dir, "mutant"),
            f"{args.prefix}_mutant",
            args,
        )

if __name__ == "__main__":
    main()
