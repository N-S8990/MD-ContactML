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
    
    logger.info("Skipping explicit solvation! Using Implicit Solvent to prevent Segfaults/OOM...")
    # fixer.addSolvent is removed.
    
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
    
    # Use amber14 forcefield with Implicit Solvent (GBn2)
    forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    
    # No PME or Cutoff for Implicit Solvent
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, 
                                     constraints=app.HBonds)
                                     
    # Langevin integrator with 1fs timestep for stability
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
    
    logger.info(f"Minimizing energy ...")
    simulation.minimizeEnergy(tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=max_min_iters)

    if seed is None:
        simulation.context.setVelocitiesToTemperature(10 * unit.kelvin)
    else:
        simulation.context.setVelocitiesToTemperature(10 * unit.kelvin, seed)
        
    logger.info("Warming up the system gently...")
    integrator.setStepSize(0.0001 * unit.picoseconds) # Extremely small timestep (0.1 fs)
    simulation.step(1000)
    
    # Heat up to target temperature
    integrator.setTemperature(temperature * unit.kelvin)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    integrator.setStepSize(0.001 * unit.picoseconds) # Normal stable timestep (1 fs)
    
    # Setup reporters
    os.makedirs(out_dir, exist_ok=True)
    dcd_path = os.path.join(out_dir, f"{prefix}_traj.dcd")
    log_path = os.path.join(out_dir, f"{prefix}_sim.log")
    
    # Save trajectory to DCD
    simulation.reporters.append(app.DCDReporter(dcd_path, max(1, report_interval)))
    
    # Save detailed stats to log file
    simulation.reporters.append(app.StateDataReporter(log_path, max(1, report_interval), step=True, 
                                                      potentialEnergy=True, temperature=True))
                                                      
    # Print beautiful progress to the terminal (stdout)
    print_interval = max(1, steps // 100) # Print progress 100 times during the run
    simulation.reporters.append(app.StateDataReporter(sys.stdout, print_interval, step=True, 
                                                      potentialEnergy=True, temperature=True, 
                                                      progress=True, remainingTime=True, 
                                                      speed=True, totalSteps=steps, separator='\t'))
                                                      
    logger.info(f"Running simulation for {steps} steps at {temperature}K...")
    simulation.step(steps)
    logger.info(f"Simulation complete. Trajectory saved to {dcd_path}")

def main():
    parser = argparse.ArgumentParser(description="Run MD Simulation with OpenMM")
    parser.add_argument("--input_pdb", type=str, default=DEFAULT_INPUT_PDB, help="Input PDB file")
    parser.add_argument("--out_dir", type=str, default="data/current_sim", help="Output directory")
    parser.add_argument("--prefix", type=str, default="sim", help="Prefix for output files")
    parser.add_argument("--steps", type=int, default=50000, help="Number of MD steps to run")
    parser.add_argument("--platform", type=str, choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], 
                        default='CPU', help="Compute platform to use")
    parser.add_argument("--max_min_iters", type=int, default=0, help="Max iterations for energy minimization (0 for unlimited)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; use a different seed for every independent replica.")
    parser.add_argument("--report_interval", type=int, default=50000,
                        help="Save one DCD frame every N steps (50,000 = 100 ps at 2 fs timestep).")
    parser.add_argument("--temperature", type=int, default=300, help="Simulation temperature in Kelvin")
    
    args = parser.parse_args()
    
    if os.path.exists(args.out_dir):
        logger.info(f"Removing older data in {args.out_dir}...")
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    prep_pdb = os.path.join(args.out_dir, f"{args.prefix}_prepared.pdb")
    
    topology, positions = prepare_system(args.input_pdb, prep_pdb)
    run_simulation(topology, positions, args.out_dir, args.prefix, args.steps, args.platform,
                   args.max_min_iters, args.seed, args.report_interval, args.temperature)

if __name__ == "__main__":
    main()
