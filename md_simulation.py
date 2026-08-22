import argparse
import logging
import os
import sys
import shutil

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
    """Uses PDBFixer to clean the structure and add missing atoms/water."""
    logger.info(f"Preparing system from {input_pdb}...")
    fixer = PDBFixer(filename=input_pdb)
    
    logger.info("Finding missing residues (and ignoring them to prevent massive box sizes)...")
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    
    logger.info("Finding nonstandard residues...")
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    logger.info("Removing heterogens (e.g., glycans, sugars) to avoid forcefield errors...")
    fixer.removeHeterogens(True)
    
    logger.info("Adding missing sidechain atoms...")
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    

    logger.info("Adding missing hydrogens...")
    fixer.addMissingHydrogens(7.0)
    
    logger.info("Solvating system...")
    fixer.addSolvent(padding=1.0 * unit.nanometers, ionicStrength=0.15 * unit.molar)
    
    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
    logger.info(f"Prepared system saved to {output_pdb}")
    return fixer.topology, fixer.positions

def run_simulation(topology, positions, out_dir, prefix, steps=10000, platform_name=None, max_min_iters=0):
    """Sets up and runs the MD simulation."""
    logger.info("Setting up simulation parameters...")
    
    # Use amber14 forcefield
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, 
                                     nonbondedCutoff=1.0*unit.nanometer,
                                     constraints=app.HBonds)
                                     
    # Langevin integrator
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    # Try to use hardware acceleration if specified or available
    if platform_name:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            simulation = app.Simulation(topology, system, integrator, platform)
            logger.info(f"Using platform: {platform_name}")
        except Exception as e:
            logger.warning(f"Failed to use platform {platform_name}: {e}. Falling back to default.")
            simulation = app.Simulation(topology, system, integrator)
    else:
        simulation = app.Simulation(topology, system, integrator)
        
    simulation.context.setPositions(positions)
    
    logger.info(f"Minimizing energy (max {max_min_iters} iterations)...")
    simulation.minimizeEnergy(maxIterations=max_min_iters)
    
    # Setup reporters
    os.makedirs(out_dir, exist_ok=True)
    dcd_path = os.path.join(out_dir, f"{prefix}_traj.dcd")
    log_path = os.path.join(out_dir, f"{prefix}_sim.log")
    
    simulation.reporters.append(app.DCDReporter(dcd_path, max(1, steps // 100)))
    simulation.reporters.append(app.StateDataReporter(log_path, max(1, steps // 10), step=True, 
                                                      potentialEnergy=True, temperature=True, volume=True))
                                                      
    logger.info(f"Running simulation for {steps} steps...")
    simulation.step(steps)
    logger.info(f"Simulation complete. Trajectory saved to {dcd_path}")

def main():
    parser = argparse.ArgumentParser(description="Run MD Simulation with OpenMM")
    parser.add_argument("--input_pdb", type=str, default=DEFAULT_INPUT_PDB, help="Input PDB file")
    parser.add_argument("--out_dir", type=str, default="data/current_sim", help="Output directory")
    parser.add_argument("--prefix", type=str, default="sim", help="Prefix for output files")
    parser.add_argument("--steps", type=int, default=1000, help="Number of MD steps to run")
    parser.add_argument("--platform", type=str, choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], 
                        default='CPU', help="Compute platform to use")
    parser.add_argument("--max_min_iters", type=int, default=0, help="Max iterations for energy minimization (0 for unlimited)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.out_dir):
        logger.info(f"Removing older data in {args.out_dir}...")
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    prep_pdb = os.path.join(args.out_dir, f"{args.prefix}_prepared.pdb")
    
    topology, positions = prepare_system(args.input_pdb, prep_pdb)
    run_simulation(topology, positions, args.out_dir, args.prefix, args.steps, args.platform, args.max_min_iters)

if __name__ == "__main__":
    main()
