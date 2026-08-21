# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///

import os
import sys
import argparse

# ========================================
# SET YOUR PDB FILE PATH HERE
# ========================================
# Change this to the path of the PDB file you want to render
DEFAULT_PDB_FILE = "/Users/nirav/Docs/MD-ContactML/6m0j.pdb" 
# ========================================

def main():
    parser = argparse.ArgumentParser(description="Render a PDB structure using PyMOL")
    # Make the argument optional (nargs='?') and use the default value defined above
    parser.add_argument("pdb_file", nargs='?', default=DEFAULT_PDB_FILE, help="Path to the PDB file to render")
    parser.add_argument("--output_name", default=None, help="Base name for the output png and pse files. If not provided, uses the PDB filename.")
    args = parser.parse_args()

    pdb_path = os.path.abspath(args.pdb_file)
    if not os.path.exists(pdb_path):
        print(f"Error: Could not find PDB file at {pdb_path}")
        sys.exit(1)

    # Use the PDB filename as the base name if none is provided
    base_name = args.output_name if args.output_name else os.path.splitext(os.path.basename(pdb_path))[0]

    # Output directory is the same as the script directory (the renders folder)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, f"{base_name}.png")
    out_pse = os.path.join(out_dir, f"{base_name}.pse")

    print("Initializing PyMOL renderer...")
    
    # Set environment variable for headless rendering
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import pymol # pytype: disable=import-error
    pymol.pymol_argv = ["pymol", "-cq"]
    pymol.finish_launching()

    from pymol import cmd # pytype: disable=import-error

    # Load the structure
    obj_name = os.path.basename(pdb_path).split('.')[0]
    cmd.load(pdb_path, obj_name)

    # Show as cartoon and color automatically by chain
    cmd.show("cartoon")
    cmd.util.color_chains("(all)")

    # Center and orient the view
    cmd.orient()
    cmd.set("ray_opaque_background", 1)

    # Save outputs
    print(f"Saving render to {out_png}")
    cmd.png(out_png, width=1200, height=900, dpi=150)
    print(f"Saving PyMOL session to {out_pse}")
    cmd.save(out_pse)
    
    cmd.quit()
    print("Done!")

if __name__ == "__main__":
    main()
