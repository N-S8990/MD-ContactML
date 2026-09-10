import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    temperatures = [300, 400]
    base_out_dir = "data"
    steps = 500000 # 500k steps as requested
    
    # Allow overriding steps for testing purposes via environment variable
    if "TEST_STEPS" in os.environ:
        steps = int(os.environ["TEST_STEPS"])

    logger.info(f"Starting temperature sweep for {temperatures} with {steps} steps each.")
    
    for temp in temperatures:
        temp_dir = os.path.join(base_out_dir, f"temp_{temp}K")
        prefix = f"sim_{temp}K"
        
        logger.info(f"--- Running simulation at {temp}K ---")
        
        cmd = [
            "python", "md_simulation.py",
            "--out_dir", temp_dir,
            "--prefix", prefix,
            "--steps", str(steps),
            "--temperature", str(temp),
            "--report_interval", "1000"
            # Default to reference or CPU for safety unless user wants CUDA
            # Given we want script to be portable for testing, we will omit platform to let openmm decide (usually CUDA if available, else CPU)
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error(f"Simulation at {temp}K failed!")
            return
            
    logger.info("Temperature sweep complete!")

if __name__ == "__main__":
    main()
