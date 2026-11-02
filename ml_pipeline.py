import time
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import networkx as nx

import MDAnalysis as mda
from MDAnalysis.analysis import distances

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, log_loss
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MD-ContactML")


class ComprehensiveFeatureExtractor:
    def __init__(self, reference_pdb):
        self.reference_pdb = reference_pdb
        self._setup_definitions()

    def _setup_definitions(self):
        logger.info(f"Configuring biophysical feature definitions from reference: {self.reference_pdb}")
        u = mda.Universe(self.reference_pdb)
        u.trajectory[0]

        chA = u.select_atoms("chainID A")
        chB = u.select_atoms("chainID B")
        heavy_A = u.select_atoms("chainID A and not name H*")
        heavy_B = u.select_atoms("chainID B and not name H*")

        d_init = distances.distance_array(heavy_A.positions, heavy_B.positions)
        close_mask = d_init < 5.0
        a_idx, b_idx = np.where(close_mask)

        self.res_A_list = sorted(list(set(heavy_A[a_idx].residues)), key=lambda r: r.resid)
        self.res_B_list = sorted(list(set(heavy_B[b_idx].residues)), key=lambda r: r.resid)

        self.candidate_pairs = []
        for rA in self.res_A_list:
            for rB in self.res_B_list:
                d_pair = distances.distance_array(rA.atoms.positions, rB.atoms.positions).min()
                if d_pair < 5.0:
                    cb_A_exists = len(rA.atoms.select_atoms("name CB")) > 0
                    cb_B_exists = len(rB.atoms.select_atoms("name CB")) > 0

                    self.candidate_pairs.append({
                        "name": f"A_{rA.resname}{rA.resid}_B_{rB.resname}{rB.resid}",
                        "rA_resid": rA.resid,
                        "rB_resid": rB.resid,
                        "rA_heavy_sel": f"chainID A and resid {rA.resid} and not name H*",
                        "rB_heavy_sel": f"chainID B and resid {rB.resid} and not name H*",
                        "ca_A_sel": f"chainID A and resid {rA.resid} and name CA",
                        "ca_B_sel": f"chainID B and resid {rB.resid} and name CA",
                        "cb_A_sel": f"chainID A and resid {rA.resid} and name CB" if cb_A_exists else f"chainID A and resid {rA.resid} and name CA",
                        "cb_B_sel": f"chainID B and resid {rB.resid} and name CB" if cb_B_exists else f"chainID B and resid {rB.resid} and name CA",
                        "sc_A_sel": f"chainID A and resid {rA.resid} and not name H* and not name N and not name C and not name O and not name CA",
                        "sc_B_sel": f"chainID B and resid {rB.resid} and not name H* and not name N and not name C and not name O and not name CA",
                    })

        logger.info(f"Interface: {len(self.res_A_list)} residues in Partner A, {len(self.res_B_list)} in Partner B.")
        logger.info(f"Configured {len(self.candidate_pairs)} candidate interface contact pairs.")

        self.sb_specs = [
            ("SB_A_LYS41_B_GLU86", "chainID A and resid 41 and name NZ", "chainID B and resid 86 and (name OE1 or name OE2)"),
            ("SB_A_ARG64_B_GLU12", "chainID A and resid 64 and (name NH1 or name NH2)", "chainID B and resid 12 and (name OE1 or name OE2)"),
            ("SB_A_ARG64_B_ASP119", "chainID A and resid 64 and (name NH1 or name NH2)", "chainID B and resid 119 and (name OD1 or name OD2)"),
            ("SB_A_ARG156_B_GLU86", "chainID A and resid 156 and (name NH1 or name NH2)", "chainID B and resid 86 and (name OE1 or name OE2)"),
            ("SB_A_ARG156_B_ASP85", "chainID A and resid 156 and (name NH1 or name NH2)", "chainID B and resid 85 and (name OD1 or name OD2)"),
            ("SB_A_GLU56_B_ARG30", "chainID A and resid 56 and (name OE1 or name OE2)", "chainID B and resid 30 and (name NH1 or name NH2)"),
            ("SB_A_ASP160_B_ARG11", "chainID A and resid 160 and (name OD1 or name OD2)", "chainID B and resid 11 and (name NH1 or name NH2)")
        ]

        self.aro_specs = [
            ("ARO_A_PHE165_B_TRP63", "chainID A and resid 165 and resname PHE", "chainID B and resid 63 and resname TRP"),
            ("ARO_A_PHE165_B_TRP124", "chainID A and resid 165 and resname PHE", "chainID B and resid 124 and resname TRP"),
            ("ARO_A_TYR42_B_TRP35", "chainID A and resid 42 and resname TYR", "chainID B and resid 35 and resname TRP"),
            ("ARO_A_TYR42_B_PHE82", "chainID A and resid 42 and resname TYR", "chainID B and resid 82 and resname PHE"),
            ("ARO_A_TYR153_B_TRP63", "chainID A and resid 153 and resname TYR", "chainID B and resid 63 and resname TRP"),
            ("ARO_A_HIS21_B_TRP63", "chainID A and resid 21 and resname HIS", "chainID B and resid 63 and resname TRP")
        ]

        self.catpi_specs = [
            ("CATPI_A_ARG64_B_TRP124", "chainID A and resid 64 and (name NH1 or name NH2 or name NE)", "chainID B and resid 124 and (name CD2 or name CE2 or name CE3 or name CZ2 or name CZ3 or name CH2)"),
            ("CATPI_A_ARG64_B_PHE82", "chainID A and resid 64 and (name NH1 or name NH2 or name NE)", "chainID B and resid 82 and (name CG or name CD1 or name CD2 or name CE1 or name CE2 or name CZ)"),
            ("CATPI_A_ARG64_B_TRP35", "chainID A and resid 64 and (name NH1 or name NH2 or name NE)", "chainID B and resid 35 and (name CD2 or name CE2 or name CE3 or name CZ2 or name CZ3 or name CH2)"),
            ("CATPI_A_LYS41_B_TRP35", "chainID A and resid 41 and name NZ", "chainID B and resid 35 and (name CD2 or name CE2 or name CE3 or name CZ2 or name CZ3 or name CH2)"),
            ("CATPI_A_LYS41_B_PHE82", "chainID A and resid 41 and name NZ", "chainID B and resid 82 and (name CG or name CD1 or name CD2 or name CE1 or name CE2 or name CZ)"),
            ("CATPI_A_ARG167_B_TRP124", "chainID A and resid 167 and (name NH1 or name NH2 or name NE)", "chainID B and resid 124 and (name CD2 or name CE2 or name CE3 or name CZ2 or name CZ3 or name CH2)"),
            ("CATPI_A_LYS168_B_TRP124", "chainID A and resid 168 and name NZ", "chainID B and resid 124 and (name CD2 or name CE2 or name CE3 or name CZ2 or name CZ3 or name CH2)"),
            ("CATPI_A_ARG156_B_TRP63", "chainID A and resid 156 and (name NH1 or name NH2 or name NE)", "chainID B and resid 63 and (name CD2 or name CE2 or name CE3 or name CZ2 or name CZ3 or name CH2)"),
            ("CATPI_B_ARG172_A_PHE165", "chainID B and resid 172 and (name NH1 or name NH2 or name NE)", "chainID A and resid 165 and (name CG or name CD1 or name CD2 or name CE1 or name CE2 or name CZ)"),
            ("CATPI_B_ARG172_A_TYR42", "chainID B and resid 172 and (name NH1 or name NH2 or name NE)", "chainID A and resid 42 and (name CG or name CD1 or name CD2 or name CE1 or name CE2 or name CZ)"),
            ("CATPI_B_ARG30_A_TYR42", "chainID B and resid 30 and (name NH1 or name NH2 or name NE)", "chainID A and resid 42 and (name CG or name CD1 or name CD2 or name CE1 or name CE2 or name CZ)"),
            ("CATPI_B_ARG11_A_TYR42", "chainID B and resid 11 and (name NH1 or name NH2 or name NE)", "chainID A and resid 42 and (name CG or name CD1 or name CD2 or name CE1 or name CE2 or name CZ)")
        ]

        self.hb_specs = [
            ("HB_A_ASN63_B_TRP124", "chainID A and resid 63 and (name OD1 or name ND2)", "chainID B and resid 124 and (name NE1 or name O)"),
            ("HB_A_SER62_B_TRP124", "chainID A and resid 62 and (name OG or name O)", "chainID B and resid 124 and (name NE1 or name O)"),
            ("HB_A_THR164_B_VAL126", "chainID A and resid 164 and (name OG1 or name O)", "chainID B and resid 126 and (name N or name O)"),
            ("HB_A_GLU163_B_ARG172", "chainID A and resid 163 and (name OE1 or name OE2)", "chainID B and resid 172 and (name NH1 or name NH2 or name NE)"),
            ("HB_A_ARG64_B_GLU12", "chainID A and resid 64 and (name NH1 or name NH2 or name NE)", "chainID B and resid 12 and (name OE1 or name OE2)"),
            ("HB_A_ARG64_B_ASP119", "chainID A and resid 64 and (name NH1 or name NH2 or name NE)", "chainID B and resid 119 and (name OD1 or name OD2)"),
            ("HB_A_GLN68_B_TRP124", "chainID A and resid 68 and (name OE1 or name NE2)", "chainID B and resid 124 and (name NE1 or name O)")
        ]

    def extract_trajectory_features(self, topology, trajectory, max_frames=None, frame_stride=1):
        logger.info(f"Extracting features from {trajectory}...")
        u = mda.Universe(topology, trajectory)

        chA = u.select_atoms("chainID A")
        chB = u.select_atoms("chainID B")
        basic_A = u.select_atoms("chainID A and resname ARG LYS HIS and (name NZ or name NH1 or name NH2 or name ND1 or name NE2)")
        acidic_B = u.select_atoms("chainID B and resname ASP GLU and (name OD1 or name OD2 or name OE1 or name OE2)")
        acidic_A = u.select_atoms("chainID A and resname ASP GLU and (name OD1 or name OD2 or name OE1 or name OE2)")
        basic_B = u.select_atoms("chainID B and resname ARG LYS HIS and (name NZ or name NH1 or name NH2 or name ND1 or name NE2)")

        hydro_A = u.select_atoms("chainID A and resname ALA VAL LEU ILE PRO MET PHE TRP and (name C* and not name C and not name CA)")
        hydro_B = u.select_atoms("chainID B and resname ALA VAL LEU ILE PRO MET PHE TRP and (name C* and not name C and not name CA)")

        polar_A = u.select_atoms("chainID A and (name O* or name N*)")
        polar_B = u.select_atoms("chainID B and (name O* or name N*)")

        sb_groups = []
        for name, selA, selB in self.sb_specs:
            gA = u.select_atoms(selA)
            gB = u.select_atoms(selB)
            if len(gA) > 0 and len(gB) > 0:
                sb_groups.append((name, gA, gB))

        aro_groups = []
        for name, selA, selB in self.aro_specs:
            gA = u.select_atoms(selA + " and not name H* and not name N and not name C and not name O")
            gB = u.select_atoms(selB + " and not name H* and not name N and not name C and not name O")
            if len(gA) > 0 and len(gB) > 0:
                aro_groups.append((name, gA, gB))

        pair_groups = []
        for pair in self.candidate_pairs:
            rA_heavy = u.select_atoms(pair["rA_heavy_sel"])
            rB_heavy = u.select_atoms(pair["rB_heavy_sel"])
            ca_A = u.select_atoms(pair["ca_A_sel"])
            ca_B = u.select_atoms(pair["ca_B_sel"])
            cb_A = u.select_atoms(pair["cb_A_sel"])
            cb_B = u.select_atoms(pair["cb_B_sel"])
            sc_A = u.select_atoms(pair["sc_A_sel"])
            sc_B = u.select_atoms(pair["sc_B_sel"])
            if len(sc_A) == 0:
                sc_A = cb_A
            if len(sc_B) == 0:
                sc_B = cb_B
            pair_groups.append({
                "name": pair["name"],
                "rA_resid": pair["rA_resid"],
                "rB_resid": pair["rB_resid"],
                "rA_heavy": rA_heavy,
                "rB_heavy": rB_heavy,
                "ca_A": ca_A,
                "ca_B": ca_B,
                "cb_A": cb_A,
                "cb_B": cb_B,
                "sc_A": sc_A,
                "sc_B": sc_B,
            })

        catpi_groups = []
        for name, selA, selB in self.catpi_specs:
            gA = u.select_atoms(selA)
            gB = u.select_atoms(selB)
            if len(gA) > 0 and len(gB) > 0:
                catpi_groups.append((name, gA, gB))

        hb_groups = []
        for name, selA, selB in self.hb_specs:
            gA = u.select_atoms(selA)
            gB = u.select_atoms(selB)
            if len(gA) > 0 and len(gB) > 0:
                hb_groups.append((name, gA, gB))

        cations_all_A = u.select_atoms("chainID A and ((resname ARG and (name NH1 or name NH2 or name NE)) or (resname LYS and name NZ))")
        aro_all_B = u.select_atoms("chainID B and resname PHE TYR TRP HIS and not name H* and not name C and not name CA and not name N and not name O and not name CB")
        cations_all_B = u.select_atoms("chainID B and ((resname ARG and (name NH1 or name NH2 or name NE)) or (resname LYS and name NZ))")
        aro_all_A = u.select_atoms("chainID A and resname PHE TYR TRP HIS and not name H* and not name C and not name CA and not name N and not name O and not name CB")

        bb_donors_A = u.select_atoms("chainID A and name N")
        bb_acceptors_B = u.select_atoms("chainID B and name O")
        bb_donors_B = u.select_atoms("chainID B and name N")
        bb_acceptors_A = u.select_atoms("chainID A and name O")

        sc_donors_A = u.select_atoms("chainID A and (name NZ or name NH1 or name NH2 or name ND2 or name NE2 or name OG or name OG1 or name OH or name NE)")
        sc_acceptors_B = u.select_atoms("chainID B and (name OD1 or name OD2 or name OE1 or name OE2 or name ND2 or name NE2 or name OG or name OG1 or name OH or name ND1)")
        sc_donors_B = u.select_atoms("chainID B and (name NZ or name NH1 or name NH2 or name ND2 or name NE2 or name OG or name OG1 or name OH or name NE)")
        sc_acceptors_A = u.select_atoms("chainID A and (name OD1 or name OD2 or name OE1 or name OE2 or name ND2 or name NE2 or name OG or name OG1 or name OH or name ND1)")

        dip_pos_A = u.select_atoms("chainID A and resname ARG LYS and (name NH1 or name NH2 or name NZ)")
        dip_neg_A = u.select_atoms("chainID A and resname ASP GLU and (name OD1 or name OD2 or name OE1 or name OE2)")
        dip_pos_B = u.select_atoms("chainID B and resname ARG LYS and (name NH1 or name NH2 or name NZ)")
        dip_neg_B = u.select_atoms("chainID B and resname ASP GLU and (name OD1 or name OD2 or name OE1 or name OE2)")

        int_res_A_str = " ".join(str(r.resid) for r in self.res_A_list)
        int_res_B_str = " ".join(str(r.resid) for r in self.res_B_list)
        heavy_int_A = u.select_atoms(f"chainID A and not name H* and resid {int_res_A_str}")
        heavy_int_B = u.select_atoms(f"chainID B and not name H* and resid {int_res_B_str}")

        torsion_groups = []
        for rA in self.res_A_list[:12]:
            res = u.select_atoms(f"chainID A and resid {rA.resid}").residues[0]
            torsion_groups.append(("A", rA.resid, res.phi_selection(), res.psi_selection(), res.chi1_selection()))

        for rB in self.res_B_list[:12]:
            res = u.select_atoms(f"chainID B and resid {rB.resid}").residues[0]
            torsion_groups.append(("B", rB.resid, res.phi_selection(), res.psi_selection(), res.chi1_selection()))

        ref_chA_pos = chA.positions.copy()
        ref_chB_pos = chB.positions.copy()

        frame_indices = list(range(0, len(u.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

        records = []
        for f_idx in frame_indices:
            u.trajectory[f_idx]
            row = {}

            row["GLOBAL_Rg_complex"] = float(u.atoms.radius_of_gyration())
            row["GLOBAL_Rg_A"] = float(chA.radius_of_gyration())
            row["GLOBAL_Rg_B"] = float(chB.radius_of_gyration())
            comA = chA.center_of_mass()
            comB = chB.center_of_mass()
            row["GLOBAL_COM_dist_AB"] = float(np.linalg.norm(comA - comB))

            inertia = u.atoms.moment_of_inertia()
            evals = np.sort(np.linalg.eigvalsh(inertia))
            row["GLOBAL_asphericity"] = float((evals[2] - 0.5 * (evals[0] + evals[1])) / (evals.sum() + 1e-6))

            row["DYN_RMSD_ChainA"] = float(np.sqrt(np.mean((chA.positions - ref_chA_pos)**2)))
            row["DYN_RMSD_ChainB"] = float(np.sqrt(np.mean((chB.positions - ref_chB_pos)**2)))

            if len(dip_pos_A) and len(dip_neg_A) and len(dip_pos_B) and len(dip_neg_B):
                pA = dip_pos_A.center_of_mass()
                nA = dip_neg_A.center_of_mass()
                pB = dip_pos_B.center_of_mass()
                nB = dip_neg_B.center_of_mass()
                vA = pA - nA
                vB = pB - nB
                normA = float(np.linalg.norm(vA))
                normB = float(np.linalg.norm(vB))
                row["DIPOLE_norm_A"] = normA
                row["DIPOLE_norm_B"] = normB
                row["DIPOLE_alignment_cos_angle"] = float(np.dot(vA, vB) / (normA * normB + 1e-9))
                row["DIPOLE_cross_dist_posA_negB"] = float(np.linalg.norm(pA - nB))
                row["DIPOLE_cross_dist_negA_posB"] = float(np.linalg.norm(nA - pB))

            if len(heavy_int_A) and len(heavy_int_B):
                d_inter = distances.distance_array(heavy_int_A.positions, heavy_int_B.positions)
                inter_cnt = int(np.sum(d_inter < 4.5))
                d_intra_A = distances.distance_array(heavy_int_A.positions, heavy_int_A.positions)
                intra_A_cnt = int((np.sum(d_intra_A < 4.5) - len(heavy_int_A)) // 2)
                d_intra_B = distances.distance_array(heavy_int_B.positions, heavy_int_B.positions)
                intra_B_cnt = int((np.sum(d_intra_B < 4.5) - len(heavy_int_B)) // 2)

                row["PACKING_inter_chain_contacts_4_5A"] = float(inter_cnt)
                row["PACKING_intra_chain_A_contacts_4_5A"] = float(intra_A_cnt)
                row["PACKING_intra_chain_B_contacts_4_5A"] = float(intra_B_cnt)
                row["PACKING_ratio_inter_to_intra"] = float(inter_cnt / (intra_A_cnt + intra_B_cnt + 1e-4))

            total_sb = 0
            if len(basic_A) and len(acidic_B):
                total_sb += np.sum(distances.distance_array(basic_A.positions, acidic_B.positions) < 4.0)
            if len(acidic_A) and len(basic_B):
                total_sb += np.sum(distances.distance_array(acidic_A.positions, basic_B.positions) < 4.0)
            row["TOTAL_salt_bridges_count"] = float(total_sb)

            for name, gA, gB in sb_groups:
                row[name] = float(distances.distance_array(gA.positions, gB.positions).min())

            for name, gA, gB in aro_groups:
                row[name] = float(distances.distance_array(gA.positions, gB.positions).min())

            catpi_tot = 0
            if len(cations_all_A) and len(aro_all_B):
                catpi_tot += int(np.sum(distances.distance_array(cations_all_A.positions, aro_all_B.positions) < 6.0))
            if len(cations_all_B) and len(aro_all_A):
                catpi_tot += int(np.sum(distances.distance_array(cations_all_B.positions, aro_all_A.positions) < 6.0))
            row["TOTAL_cation_pi_contacts_6_0A"] = float(catpi_tot)

            for name, gA, gB in catpi_groups:
                d_val = float(distances.distance_array(gA.positions, gB.positions).min())
                row[f"{name}_dist"] = d_val
                row[f"{name}_soft"] = float(np.clip((1.0 - (d_val / 6.0)**6) / (1.0 - (d_val / 6.0)**12 + 1e-9) if d_val != 6.0 else 0.5, 0.0, 1.0))
                row[f"{name}_inv"] = float(1.0 / (d_val + 1e-2))

            for name, gA, gB in hb_groups:
                row[name] = float(distances.distance_array(gA.positions, gB.positions).min())

            d_bb1 = distances.distance_array(bb_donors_A.positions, bb_acceptors_B.positions) if len(bb_donors_A) and len(bb_acceptors_B) else np.array([])
            d_bb2 = distances.distance_array(bb_donors_B.positions, bb_acceptors_A.positions) if len(bb_donors_B) and len(bb_acceptors_A) else np.array([])
            cnt_bb_35 = (np.sum(d_bb1 < 3.5) if d_bb1.size else 0) + (np.sum(d_bb2 < 3.5) if d_bb2.size else 0)
            cnt_bb_30 = (np.sum(d_bb1 < 3.0) if d_bb1.size else 0) + (np.sum(d_bb2 < 3.0) if d_bb2.size else 0)
            row["HBOND_BB_BB_3_5A"] = float(cnt_bb_35)
            row["HBOND_BB_BB_3_0A"] = float(cnt_bb_30)

            d_sc1 = distances.distance_array(sc_donors_A.positions, sc_acceptors_B.positions) if len(sc_donors_A) and len(sc_acceptors_B) else np.array([])
            d_sc2 = distances.distance_array(sc_donors_B.positions, sc_acceptors_A.positions) if len(sc_donors_B) and len(sc_acceptors_A) else np.array([])
            cnt_sc_35 = (np.sum(d_sc1 < 3.5) if d_sc1.size else 0) + (np.sum(d_sc2 < 3.5) if d_sc2.size else 0)
            cnt_sc_30 = (np.sum(d_sc1 < 3.0) if d_sc1.size else 0) + (np.sum(d_sc2 < 3.0) if d_sc2.size else 0)
            row["HBOND_SC_SC_3_5A"] = float(cnt_sc_35)
            row["HBOND_SC_SC_3_0A"] = float(cnt_sc_30)

            d_mix1 = distances.distance_array(sc_donors_A.positions, bb_acceptors_B.positions) if len(sc_donors_A) and len(bb_acceptors_B) else np.array([])
            d_mix2 = distances.distance_array(bb_donors_A.positions, sc_acceptors_B.positions) if len(bb_donors_A) and len(sc_acceptors_B) else np.array([])
            d_mix3 = distances.distance_array(sc_donors_B.positions, bb_acceptors_A.positions) if len(sc_donors_B) and len(bb_acceptors_A) else np.array([])
            d_mix4 = distances.distance_array(bb_donors_B.positions, sc_acceptors_A.positions) if len(bb_donors_B) and len(sc_acceptors_A) else np.array([])
            cnt_mix_35 = sum(np.sum(x < 3.5) for x in [d_mix1, d_mix2, d_mix3, d_mix4] if x.size)
            cnt_mix_30 = sum(np.sum(x < 3.0) for x in [d_mix1, d_mix2, d_mix3, d_mix4] if x.size)
            row["HBOND_SC_BB_3_5A"] = float(cnt_mix_35)
            row["HBOND_SC_BB_3_0A"] = float(cnt_mix_30)
            row["HBOND_total_3_5A"] = float(cnt_bb_35 + cnt_sc_35 + cnt_mix_35)
            row["HBOND_total_3_0A"] = float(cnt_bb_30 + cnt_sc_30 + cnt_mix_30)

            if len(hydro_A) and len(hydro_B):
                row["TOTAL_hydrophobic_contacts"] = float(np.sum(distances.distance_array(hydro_A.positions, hydro_B.positions) < 4.5))
            else:
                row["TOTAL_hydrophobic_contacts"] = 0.0

            if len(polar_A) and len(polar_B):
                d_pol = distances.distance_array(polar_A.positions, polar_B.positions)
                row["TOTAL_polar_contacts_3_5A"] = float(np.sum(d_pol < 3.5))
                row["TOTAL_polar_contacts_3_0A"] = float(np.sum(d_pol < 3.0))
            else:
                row["TOTAL_polar_contacts_3_5A"] = 0.0
                row["TOTAL_polar_contacts_3_0A"] = 0.0

            row["RATIO_polar_to_hydrophobic"] = float(row["TOTAL_polar_contacts_3_5A"] / (row["TOTAL_hydrophobic_contacts"] + 1e-4))

            pair_d_mins = []
            for pair in pair_groups:
                name = pair["name"]
                dmin = float(distances.distance_array(pair["rA_heavy"].positions, pair["rB_heavy"].positions).min())
                row[f"DMIN_{name}"] = dmin
                pair_d_mins.append((pair["rA_resid"], pair["rB_resid"], dmin))

                d_ca = float(np.linalg.norm(pair["ca_A"].positions[0] - pair["ca_B"].positions[0]))
                row[f"DCA_{name}"] = d_ca

                d_cb = float(np.linalg.norm(pair["cb_A"].positions[0] - pair["cb_B"].positions[0]))
                row[f"DCB_{name}"] = d_cb

                d_sc = float(np.linalg.norm(pair["sc_A"].center_of_mass() - pair["sc_B"].center_of_mass()))
                row[f"DCOM_SC_{name}"] = d_sc

                r_ratio = dmin / 5.0
                s_val = (1.0 - (r_ratio)**6) / (1.0 - (r_ratio)**12 + 1e-9) if r_ratio != 1.0 else 0.5
                row[f"SOFT_{name}"] = float(np.clip(s_val, 0.0, 1.0))
                row[f"INV_{name}"] = float(1.0 / (dmin + 1e-2))

            G = nx.Graph()
            for rA in self.res_A_list:
                G.add_node(f"A_{rA.resid}")
            for rB in self.res_B_list:
                G.add_node(f"B_{rB.resid}")

            edge_cnt = 0
            for rA_id, rB_id, d_val in pair_d_mins:
                if d_val < 4.5:
                    G.add_edge(f"A_{rA_id}", f"B_{rB_id}")
                    edge_cnt += 1

            degrees = [deg for n, deg in G.degree()]
            row["GRAPH_total_edges"] = float(edge_cnt)
            row["GRAPH_avg_degree"] = float(np.mean(degrees)) if degrees else 0.0
            row["GRAPH_max_degree"] = float(max(degrees)) if degrees else 0.0

            deg_cent = nx.degree_centrality(G)
            row["GRAPH_avg_degree_centrality"] = float(np.mean(list(deg_cent.values())))

            betweenness = nx.betweenness_centrality(G)
            row["GRAPH_max_betweenness"] = float(max(betweenness.values())) if betweenness else 0.0
            row["GRAPH_avg_betweenness"] = float(np.mean(list(betweenness.values()))) if betweenness else 0.0

            row["GRAPH_clustering_coefficient"] = float(nx.average_clustering(G))
            row["GRAPH_transitivity"] = float(nx.transitivity(G))
            row["GRAPH_connected_components"] = float(nx.number_connected_components(G))
            if len(G) > 1:
                L = nx.normalized_laplacian_matrix(G).toarray()
                evals_L = np.sort(np.linalg.eigvalsh(L))
                row["GRAPH_spectral_gap_lambda2"] = float(evals_L[1]) if len(evals_L) > 1 else 0.0
            else:
                row["GRAPH_spectral_gap_lambda2"] = 0.0

            for chain_tag, resid, phi, psi, chi1 in torsion_groups:
                if phi is not None:
                    phi_rad = np.radians(phi.dihedral.value())
                    row[f"TORSION_{chain_tag}_{resid}_sin_phi"] = float(np.sin(phi_rad))
                    row[f"TORSION_{chain_tag}_{resid}_cos_phi"] = float(np.cos(phi_rad))
                if psi is not None:
                    psi_rad = np.radians(psi.dihedral.value())
                    row[f"TORSION_{chain_tag}_{resid}_sin_psi"] = float(np.sin(psi_rad))
                    row[f"TORSION_{chain_tag}_{resid}_cos_psi"] = float(np.cos(psi_rad))
                if chi1 is not None:
                    chi1_rad = np.radians(chi1.dihedral.value())
                    row[f"TORSION_{chain_tag}_{resid}_sin_chi1"] = float(np.sin(chi1_rad))
                    row[f"TORSION_{chain_tag}_{resid}_cos_chi1"] = float(np.cos(chi1_rad))

            records.append(row)

        df = pd.DataFrame(records)
        return df

    def build_dataset(self, class0_top, class0_trajs, class1_top, class1_trajs, max_frames=None, frame_stride=1):
        frames = []
        labels = []
        groups = []

        for i, traj in enumerate(class0_trajs):
            rep_label = f"rep_{i+1}"
            df_rep = self.extract_trajectory_features(class0_top, traj, max_frames=max_frames, frame_stride=frame_stride)
            frames.append(df_rep)
            labels.extend([0] * len(df_rep))
            groups.extend([rep_label] * len(df_rep))

        for i, traj in enumerate(class1_trajs):
            rep_label = f"rep_{i+1}"
            df_rep = self.extract_trajectory_features(class1_top, traj, max_frames=max_frames, frame_stride=frame_stride)
            frames.append(df_rep)
            labels.extend([1] * len(df_rep))
            groups.extend([rep_label] * len(df_rep))

        X_full = pd.concat(frames, ignore_index=True)
        y_full = np.array(labels)
        groups_full = np.array(groups)

        logger.info(f"Dataset successfully compiled: {X_full.shape[0]} frames, {X_full.shape[1]} features.")
        return X_full, y_full, groups_full


class MDPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.active_cols = None

    def fit_transform_split(self, X_full, y_full, groups_full, test_group="rep_3"):
        train_mask = (groups_full != test_group)
        test_mask = (groups_full == test_group)

        X_train_raw = X_full[train_mask].reset_index(drop=True)
        y_train = y_full[train_mask]
        groups_train = groups_full[train_mask]

        X_test_raw = X_full[test_mask].reset_index(drop=True)
        y_test = y_full[test_mask]

        var_filter = X_train_raw.std() > 1e-6
        self.active_cols = X_train_raw.columns[var_filter].tolist()
        logger.info(f"Filtered out {X_full.shape[1] - len(self.active_cols)} constant features. Active features: {len(self.active_cols)}")

        X_tr_clean = X_train_raw[self.active_cols]
        X_te_clean = X_test_raw[self.active_cols]

        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_tr_clean), columns=self.active_cols)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_te_clean), columns=self.active_cols)

        return X_train_scaled, y_train, groups_train, X_test_scaled, y_test


class CorrelationAnalyzer:
    def __init__(self, threshold=0.98):
        self.threshold = threshold
        self.corr_matrix = None

    def fit(self, X):
        logger.info(f"Computing full Pearson correlation matrix for {X.shape[1]} features...")
        self.corr_matrix = X.corr().abs()
        return self.corr_matrix

    def get_features_to_drop_batch(self, active_features, batch_size=25, feature_importances=None):
        sub_corr = self.corr_matrix.loc[active_features, active_features]
        upper = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))

        high_corr_mask = (upper >= self.threshold)
        if not high_corr_mask.any().any():
            return [], 0.0

        pairs = np.argwhere(high_corr_mask.values)
        corrs = [upper.iloc[r, c] for r, c in pairs]
        sorted_idx = np.argsort(corrs)[::-1]
        max_corr_step = corrs[sorted_idx[0]]

        features_to_drop = set()
        for idx in sorted_idx:
            r, c = pairs[idx]
            feat_A = upper.index[r]
            feat_B = upper.columns[c]

            if feat_A not in features_to_drop and feat_B not in features_to_drop:
                imp_A = feature_importances.get(feat_A, 0.0) if feature_importances else 0.0
                imp_B = feature_importances.get(feat_B, 0.0) if feature_importances else 0.0
                if imp_A >= imp_B:
                    features_to_drop.add(feat_B)
                else:
                    features_to_drop.add(feat_A)

            if len(features_to_drop) >= batch_size:
                break

        return list(features_to_drop), float(max_corr_step)


class FeatureEliminationLoop:
    def __init__(self, corr_threshold=0.98, accuracy_tolerance=0.05, out_dir="results"):
        self.analyzer = CorrelationAnalyzer(threshold=corr_threshold)
        self.accuracy_tolerance = accuracy_tolerance
        self.out_dir = Path(out_dir)
        self.corr_dir = self.out_dir / "correlation_matrices"
        self.corr_dir.mkdir(parents=True, exist_ok=True)

    def run(self, X_train, y_train, groups_train, X_test, y_test):
        logger.info(f"Starting Iterative Feature Elimination at threshold = {self.analyzer.threshold}...")

        val_mask = (groups_train == "rep_2")
        train_mask = (groups_train == "rep_1")

        X_tr_inner = X_train.loc[train_mask].reset_index(drop=True)
        y_tr_inner = y_train[train_mask]
        X_val_inner = X_train.loc[val_mask].reset_index(drop=True)
        y_val_inner = y_train[val_mask]

        active_features = list(X_train.columns)

        corr_initial = self.analyzer.fit(X_train)
        corr_initial.to_csv(self.corr_dir / "corr_matrix_initial.csv")

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr_inner, y_tr_inner)
        baseline_acc = accuracy_score(y_val_inner, rf.predict(X_val_inner))
        importances = dict(zip(active_features, rf.feature_importances_))

        log = [{
            "iteration": 0,
            "action": "Baseline",
            "feature_count": len(active_features),
            "max_correlation": float(corr_initial.values[np.triu_indices_from(corr_initial, k=1)].max()),
            "val_accuracy": baseline_acc
        }]

        iteration = 1
        pbar = tqdm(desc="Eliminating Collinear Features", unit="iter")

        while True:
            features_to_drop, max_corr = self.analyzer.get_features_to_drop_batch(
                active_features, batch_size=25, feature_importances=importances
            )

            if not features_to_drop:
                logger.info(f"Stopping: No feature pairs with |r| >= {self.analyzer.threshold} remain.")
                break

            candidate_features = [f for f in active_features if f not in features_to_drop]

            rf.fit(X_tr_inner[candidate_features], y_tr_inner)
            current_acc = accuracy_score(y_val_inner, rf.predict(X_val_inner[candidate_features]))

            pbar.set_postfix({"feats": len(candidate_features), "acc": f"{current_acc:.3f}", "max_r": f"{max_corr:.3f}"})
            pbar.update(1)

            if (baseline_acc - current_acc) > self.accuracy_tolerance:
                logger.warning(f"Stopping: Accuracy dropped below tolerance ({current_acc:.4f} < {baseline_acc - self.accuracy_tolerance:.4f}). Reverting last step.")
                break

            active_features = candidate_features
            log.append({
                "iteration": iteration,
                "action": f"Dropped {len(features_to_drop)} features",
                "feature_count": len(active_features),
                "max_correlation": float(max_corr),
                "val_accuracy": float(current_acc)
            })

            iteration += 1

        pbar.close()
        logger.info(f"Elimination complete. {len(active_features)} non-redundant features retained.")

        corr_final = X_train[active_features].corr().abs()
        corr_final.to_csv(self.corr_dir / "corr_matrix_final.csv")

        log_df = pd.DataFrame(log)
        log_df.to_csv(self.out_dir / "elimination_log.csv", index=False)

        return active_features, log_df, corr_initial, corr_final


class ModelTrainer:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }

    def train_and_evaluate(self, X_tr, y_tr, X_te, y_te):
        metrics = {}
        for name, model in self.models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

            acc = accuracy_score(y_te, preds)
            bal_acc = balanced_accuracy_score(y_te, preds)
            prec = precision_score(y_te, preds, zero_division=0)
            rec = recall_score(y_te, preds)
            f1 = f1_score(y_te, preds)
            auc = roc_auc_score(y_te, probs) if probs is not None else float('nan')

            metrics[name] = {
                "Accuracy": acc,
                "Balanced Accuracy": bal_acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc
            }
            logger.info(f"[{name}] Test Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        return pd.DataFrame(metrics).T


def categorize_feature(feat):
    if feat.startswith("SB_") or feat.startswith("TOTAL_salt"):
        return "Salt Bridge / Electrostatic"
    elif feat.startswith("ARO_"):
        return "Aromatic / Pi-Stacking"
    elif feat.startswith("CATPI_") or feat.startswith("TOTAL_cation_pi"):
        return "Cation-Pi Interaction"
    elif feat.startswith("HBOND_") or feat.startswith("HB_"):
        return "Directional Hydrogen Bond"
    elif feat.startswith("DIPOLE_"):
        return "Dipole & Electrostatic Alignment"
    elif feat.startswith("PACKING_"):
        return "Interface Packing Density"
    elif feat.startswith("TOTAL_hydrophobic") or feat.startswith("RATIO_polar"):
        return "Hydrophobic Packing"
    elif feat.startswith("TOTAL_polar"):
        return "Hydrogen Bonding / Polar"
    elif feat.startswith("GRAPH_"):
        return "Network Topology / Centrality"
    elif feat.startswith("GLOBAL_"):
        return "Global Shape & Size"
    elif feat.startswith("DYN_"):
        return "Dynamics & Fluctuation"
    elif feat.startswith("TORSION_"):
        return "Backbone / Sidechain Dihedral"
    elif feat.startswith("DCA_"):
        return "C-alpha Distance"
    elif feat.startswith("DCB_"):
        return "C-beta Residue Distance"
    elif feat.startswith("DMIN_"):
        return "Minimum Heavy-Atom Contact"
    elif feat.startswith("DCOM_SC_"):
        return "Sidechain Centroid Packing"
    elif feat.startswith("SOFT_"):
        return "Soft Contact Probability"
    elif feat.startswith("INV_"):
        return "Inverse Contact Proximity"
    else:
        return "Biophysical"


class LOFOAblationEngine:
    def __init__(self, out_dir="results"):
        self.out_dir = Path(out_dir)

    def run(self, X_train, y_train, X_test, y_test, max_corr_series=None):
        logger.info(f"Starting Leave-One-Feature-Out (LOFO) ablation across {X_train.shape[1]} features...")

        baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        baseline_clf.fit(X_train, y_train)

        base_preds = baseline_clf.predict(X_test)
        base_probs = baseline_clf.predict_proba(X_test)

        base_acc = accuracy_score(y_test, base_preds)
        base_f1 = f1_score(y_test, base_preds)
        base_auc = roc_auc_score(y_test, base_probs[:, 1])
        base_loss = log_loss(y_test, base_probs)

        logger.info(f"LOFO Baseline Test Accuracy: {base_acc:.4f} | Loss: {base_loss:.4f}")

        records = []
        pbar = tqdm(total=X_train.shape[1], desc="LOFO Ablation Testing", unit="feat")

        for feat in X_train.columns:
            sub_cols = [c for c in X_train.columns if c != feat]
            X_tr_sub = X_train[sub_cols]
            X_te_sub = X_test[sub_cols]

            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            clf.fit(X_tr_sub, y_train)

            preds = clf.predict(X_te_sub)
            probs = clf.predict_proba(X_te_sub)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_test, probs[:, 1])
            except Exception:
                auc = float('nan')
            try:
                loss = log_loss(y_test, probs)
            except Exception:
                loss = float('nan')

            delta_acc = base_acc - acc
            delta_loss = loss - base_loss

            records.append({
                "Feature": feat,
                "Category": categorize_feature(feat),
                "Accuracy_Without_Feature": acc,
                "Accuracy_Drop": delta_acc,
                "F1_Drop": base_f1 - f1,
                "AUC_Drop": base_auc - auc,
                "Loss_Increase": delta_loss,
                "Max_Correlation_With_Others": float(max_corr_series.get(feat, 0.0)) if max_corr_series is not None else float('nan')
            })

            pbar.set_postfix({"feat": feat[:16], "d_acc": f"{delta_acc:+.3f}"})
            pbar.update(1)

        pbar.close()

        lofo_df = pd.DataFrame(records)
        lofo_df = lofo_df.sort_values(by=["Accuracy_Drop", "Loss_Increase"], ascending=[False, False]).reset_index(drop=True)
        lofo_df["Rank"] = lofo_df.index + 1

        lofo_df.to_csv(self.out_dir / "lofo_feature_importance.csv", index=False)
        return lofo_df, {
            "baseline_accuracy": base_acc,
            "baseline_f1": base_f1,
            "baseline_auc": base_auc,
            "baseline_loss": base_loss
        }


def map_feature_to_cluster(feat):
    m_pair = re.search(r'([A-Z0-9]+)_A_([A-Z0-9]+)_B_([A-Z0-9]+)', feat)
    if m_pair:
        return f"PAIR_A_{m_pair.group(2)}_B_{m_pair.group(3)}"
    m_pair_rev = re.search(r'([A-Z0-9]+)_B_([A-Z0-9]+)_A_([A-Z0-9]+)', feat)
    if m_pair_rev:
        return f"PAIR_A_{m_pair_rev.group(3)}_B_{m_pair_rev.group(2)}"
    if feat.startswith("GLOBAL_"):
        return "CLUSTER_GLOBAL_SHAPE"
    if feat.startswith("DYN_"):
        return "CLUSTER_GLOBAL_DYNAMICS"
    if feat.startswith("DIPOLE_"):
        return "CLUSTER_ELECTROSTATIC_DIPOLE"
    if feat.startswith("PACKING_") or feat.startswith("TOTAL_hydrophobic") or feat.startswith("RATIO_polar"):
        return "CLUSTER_INTERFACE_PACKING"
    if feat.startswith("HBOND_") or feat.startswith("TOTAL_polar"):
        return "CLUSTER_DIRECTIONAL_HBONDS"
    if feat.startswith("GRAPH_"):
        return "CLUSTER_GRAPH_TOPOLOGY"
    if feat.startswith("TORSION_"):
        m = re.match(r"(TORSION_[AB]_\d+)", feat)
        if m:
            return m.group(1)
        return "CLUSTER_TORSIONS"
    if feat.startswith("TOTAL_salt"):
        return "CLUSTER_SALT_BRIDGES_GENERAL"
    if feat.startswith("TOTAL_cation_pi"):
        return "CLUSTER_CATION_PI_GENERAL"
    return "CLUSTER_MISC"


def categorize_cluster(cluster):
    if cluster.startswith("PAIR_"):
        return "Residue-Pair Interface Contact"
    elif cluster == "CLUSTER_GLOBAL_SHAPE":
        return "Global Shape & Size"
    elif cluster == "CLUSTER_GLOBAL_DYNAMICS":
        return "Global Dynamics & RMSD"
    elif cluster == "CLUSTER_ELECTROSTATIC_DIPOLE":
        return "Dipole & Electrostatic Alignment"
    elif cluster == "CLUSTER_INTERFACE_PACKING":
        return "Interface Packing Density"
    elif cluster == "CLUSTER_DIRECTIONAL_HBONDS":
        return "Directional Hydrogen Bonds"
    elif cluster == "CLUSTER_GRAPH_TOPOLOGY":
        return "Network Graph Topology"
    elif cluster.startswith("TORSION_"):
        return "Backbone / Sidechain Dihedral"
    elif cluster == "CLUSTER_CATION_PI_GENERAL":
        return "Cation-Pi Interaction"
    elif cluster == "CLUSTER_SALT_BRIDGES_GENERAL":
        return "Salt Bridge / Electrostatic"
    else:
        return "Biophysical Cluster"


class GroupLOFOEngine:
    def __init__(self, out_dir="results"):
        self.out_dir = Path(out_dir)

    def run(self, X_train, y_train, X_test, y_test, prom_df=None):
        logger.info("Starting Group-Level LOFO (Cluster Ablation) across functional & residue clusters...")
        cluster_map = {c: map_feature_to_cluster(c) for c in X_train.columns}
        clusters = sorted(list(set(cluster_map.values())))
        logger.info(f"Identified {len(clusters)} distinct biological clusters across {X_train.shape[1]} features.")

        baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        baseline_clf.fit(X_train, y_train)

        base_preds = baseline_clf.predict(X_test)
        base_probs = baseline_clf.predict_proba(X_test)

        base_acc = accuracy_score(y_test, base_preds)
        base_f1 = f1_score(y_test, base_preds)
        base_auc = roc_auc_score(y_test, base_probs[:, 1])
        base_loss = log_loss(y_test, base_probs)

        logger.info(f"Group LOFO Baseline Accuracy: {base_acc:.4f} | Loss: {base_loss:.4f}")

        prom_dict = {}
        if prom_df is not None:
            for _, r in prom_df.iterrows():
                prom_dict[r["Feature"]] = r["Prominence_Score"]

        records = []
        pbar = tqdm(total=len(clusters), desc="Cluster Ablation Testing", unit="clust")

        for cl in clusters:
            sub_cols = [c for c in X_train.columns if cluster_map[c] != cl]
            cl_feats = [c for c in X_train.columns if cluster_map[c] == cl]

            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train[sub_cols], y_train)

            preds = clf.predict(X_test[sub_cols])
            probs = clf.predict_proba(X_test[sub_cols])

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_test, probs[:, 1])
            except Exception:
                auc = float('nan')
            try:
                loss = log_loss(y_test, probs)
            except Exception:
                loss = float('nan')

            delta_acc = base_acc - acc
            delta_loss = loss - base_loss

            cl_prom_scores = [prom_dict.get(f, 0.0) for f in cl_feats]
            sum_prom = sum(cl_prom_scores)
            max_prom = max(cl_prom_scores) if cl_prom_scores else 0.0

            records.append({
                "Cluster": cl,
                "Category": categorize_cluster(cl),
                "Num_Features": len(cl_feats),
                "Sample_Features": ", ".join(cl_feats[:3]) + (f" (+{len(cl_feats)-3} more)" if len(cl_feats) > 3 else ""),
                "Accuracy_Without_Cluster": acc,
                "Accuracy_Drop": delta_acc,
                "Loss_Increase": delta_loss,
                "F1_Drop": base_f1 - f1,
                "AUC_Drop": base_auc - auc,
                "Max_Prominence_Score": max_prom,
                "Sum_Prominence_Score": sum_prom
            })

            pbar.set_postfix({"cl": cl[:18], "d_acc": f"{delta_acc:+.3f}"})
            pbar.update(1)

        pbar.close()

        cl_df = pd.DataFrame(records)
        max_d_acc = max(cl_df["Accuracy_Drop"].max(), 1e-4)
        max_prom_all = max(cl_df["Max_Prominence_Score"].max(), 1e-4)
        cl_df["Master_Impact_Score"] = 0.5 * (np.clip(cl_df["Accuracy_Drop"], 0, None) / max_d_acc) + \
                                       0.5 * (cl_df["Max_Prominence_Score"] / max_prom_all)

        cl_df = cl_df.sort_values(by=["Accuracy_Drop", "Master_Impact_Score", "Loss_Increase"], ascending=[False, False, False]).reset_index(drop=True)
        cl_df["Rank"] = cl_df.index + 1

        cl_df.to_csv(self.out_dir / "cluster_ablation_importance.csv", index=False)
        return cl_df, {
            "baseline_accuracy": base_acc,
            "baseline_f1": base_f1,
            "baseline_auc": base_auc,
            "baseline_loss": base_loss
        }


class PipelineVisualizer:
    def __init__(self, out_dir="results"):
        self.figures_dir = Path(out_dir) / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_correlation_heatmap(self, corr_df, title, filename):
        plt.figure(figsize=(10, 8))
        if corr_df.shape[0] > 70:
            sample_idx = np.linspace(0, corr_df.shape[0] - 1, 70, dtype=int)
            plot_df = corr_df.iloc[sample_idx, sample_idx]
        else:
            plot_df = corr_df

        sns.heatmap(plot_df, cmap="viridis", vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        plt.title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300)
        plt.close()

    def plot_elimination_trajectory(self, log_df, filename="elimination_trajectory.png"):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Elimination Iteration', fontsize=12, fontweight="bold")
        ax1.set_ylabel('Surviving Feature Count', color=color, fontsize=12, fontweight="bold")
        ax1.plot(log_df['iteration'], log_df['feature_count'], color=color, marker='o', linewidth=2, label="Feature Count")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Validation Accuracy', color=color, fontsize=12, fontweight="bold")
        ax2.plot(log_df['iteration'], log_df['val_accuracy'], color=color, marker='s', linewidth=2, linestyle='--', label="Accuracy")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0.80, 1.02])

        plt.title("Iterative Collinearity Elimination (Threshold = 0.98)", fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300)
        plt.close()

    def plot_prominent_features(self, prom_df, filename="top_prominent_features.png", top_n=25):
        top_df = prom_df.head(top_n)
        plt.figure(figsize=(12, 10))
        palette = sns.color_palette("tab10", len(top_df["Category"].unique()))
        cat_colors = dict(zip(top_df["Category"].unique(), palette))
        colors = [cat_colors[cat] for cat in top_df["Category"]]

        plt.barh(range(len(top_df)), top_df["Prominence_Score"], color=colors, edgecolor="black", alpha=0.85)
        plt.yticks(range(len(top_df)), top_df["Feature"], fontsize=9, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.xlabel("Prominence Score (RF Importance & Permutation Score)", fontsize=11, fontweight="bold")
        plt.title(f"Top {top_n} Most Prominent Biophysical Features", fontsize=14, fontweight="bold", pad=14)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=col, edgecolor='black', label=cat) for cat, col in cat_colors.items()]
        plt.legend(handles=legend_elements, title="Biophysical Category", loc="lower right", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300)
        plt.close()

    def plot_lofo_accuracy_drop(self, lofo_df, filename="lofo_top_features_accuracy_drop.png", top_n=None, min_drop=0.002):
        if min_drop is not None:
            top_df = lofo_df[lofo_df["Accuracy_Drop"] >= min_drop]
            if len(top_df) == 0:
                top_df = lofo_df.head(top_n or 25)
        elif top_n is not None:
            top_df = lofo_df.head(top_n)
        else:
            top_df = lofo_df[lofo_df["Accuracy_Drop"] > 0]

        fig_height = max(8, len(top_df) * 0.32)
        plt.figure(figsize=(12, fig_height))
        palette = sns.color_palette("tab10", len(top_df["Category"].unique()))
        cat_colors = dict(zip(top_df["Category"].unique(), palette))
        colors = [cat_colors[cat] for cat in top_df["Category"]]

        plt.barh(range(len(top_df)), top_df["Accuracy_Drop"] * 100, color=colors, edgecolor="black", alpha=0.85)
        plt.yticks(range(len(top_df)), [f"{f} [{c}]" for f, c in zip(top_df["Feature"], top_df["Category"])], fontsize=8.5, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.xlabel("Test Accuracy Drop When Feature is Removed (% points)", fontsize=11, fontweight="bold")
        plt.title(f"LOFO Ablation: All {len(top_df)} Features with Accuracy Drop >= {min_drop*100:.2f}%", fontsize=13, fontweight="bold", pad=14)

        legend_elements = [Patch(facecolor=col, edgecolor='black', label=cat) for cat, col in cat_colors.items()]
        plt.legend(handles=legend_elements, title="Biophysical Category", loc="lower right", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300)
        plt.close()

    def plot_cluster_ablation(self, cluster_df, filename="cluster_ablation_accuracy_drop.png", top_n=None, min_drop=0.002):
        if min_drop is not None:
            plot_df = cluster_df[cluster_df["Accuracy_Drop"] >= min_drop]
            if len(plot_df) == 0:
                plot_df = cluster_df.head(top_n or 25)
        elif top_n is not None:
            plot_df = cluster_df.head(top_n)
        else:
            plot_df = cluster_df[cluster_df["Accuracy_Drop"] > 0]

        fig_height = max(8, len(plot_df) * 0.35)
        plt.figure(figsize=(13, fig_height))
        palette = sns.color_palette("tab10", len(plot_df["Category"].unique()))
        cat_colors = dict(zip(plot_df["Category"].unique(), palette))
        colors = [cat_colors[cat] for cat in plot_df["Category"]]

        plt.barh(range(len(plot_df)), plot_df["Accuracy_Drop"] * 100, color=colors, edgecolor="black", alpha=0.85)
        labels = [f"{c} [{k}] ({n} feats)" for c, k, n in zip(plot_df["Cluster"], plot_df["Category"], plot_df["Num_Features"])]
        plt.yticks(range(len(plot_df)), labels, fontsize=8.5, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.xlabel("Test Accuracy Drop When Entire Cluster is Removed (% points)", fontsize=11, fontweight="bold")
        plt.title(f"Group-Level LOFO: All {len(plot_df)} Functional Clusters with Accuracy Drop >= {min_drop*100:.2f}%", fontsize=13, fontweight="bold", pad=14)

        legend_elements = [Patch(facecolor=col, edgecolor='black', label=cat) for cat, col in cat_colors.items()]
        plt.legend(handles=legend_elements, title="Cluster Category", loc="lower right", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300)
        plt.close()
