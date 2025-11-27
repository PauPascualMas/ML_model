import pandas as pd
import numpy as np

def classify_cancer(code):
    if pd.isna(code):
        return np.nan

    code = code.upper().strip()

    # -----------------------------------------
    # BENIGN AND UNCERTAIN
    if code.startswith("D"):
        return "Benign/Uncertain"

    # -----------------------------------------
    # SPECIAL PREFIX CODES (non-numeric)
    # -----------------------------------------
    if code.startswith("C7A"):
        return "Endocrine"
    if code.startswith("C7B"):
        return "Endocrine"
    if code.startswith("C4A"):
        return "Skin/Soft_Tissue"

    # -----------------------------------------
    # Extract numeric part (e.g., C43 → 43)
    # -----------------------------------------
    try:
        num = int(''.join(filter(str.isdigit, code)))
    except ValueError:
        return "Other"

    # -----------------------------------------
    # ICD-10 C-CODE CLASSIFICATION
    # -----------------------------------------S

    # C00–C14 — Lip, oral cavity, pharynx
    if 0 <= num <= 14:
        return "Oral"

    # C15–C26 — Digestive organs
    if 15 <= num <= 26:
        return "Respiratory/Digestive"

    # C30–C39 — Respiratory system & intrathoracic organs
    if 30 <= num <= 39:
        return "Respiratory/Digestive"

    # C40–C41 — Bone & articular cartilage
    if 40 <= num <= 41:
        return "Bones/Cartilage"

    # C43–C44 — Skin (melanoma, Merkel cell, other)
    if 43 <= num <= 44:
        return "Skin/Soft_Tissue"

    # C45 — Mesothelioma
    if num == 45:
        return "Skin/Soft_Tissue"

    # C46 — Kaposi sarcoma
    if num == 46:
        return "Hematopoietic/Lymphoid"

    # C47–C49 — Peripheral nerves, retro-/peritoneum, soft tissues
    if 47 <= num <= 49:
        return "Skin/Soft_Tissue"

    # C50–C58 — Breast + female genital organs
    if num==50:
        return "Breast"
    #
    if 51 <= num <= 68:
        return "Genitourinary"

    # C69–C72 — Eye, brain, CNS
    if 69 <= num <= 72:
        return "Nervous_System"

    # C73–C76 — Thyroid, endocrine glands, and ill-defined
    if 73 <= num < 77:
        return "Endocrine"

    # C77 — Secondary malignant neoplasm of lymph nodes
    if num == 77:
        return "Hematopoietic/Lymphoid"

    # C78–C80 — Secondary cancers (respiratory, digestive, unspecified)
    if 78 <= num < 80:
        return "Respiratory/Digestive"

    # C80 specifically – unspecified malignant neoplasm
    if num == 80:
        return "Other"

    # C81–C96 — Lymphoid, hematopoietic, related tissue
    if 81 <= num <= 96:
        return "Hematopoietic/Lymphoid"

    # -----------------------------------------
    # EVERYTHING ELSE
    # -----------------------------------------
    return "Other"


# genera belonging to Enterobacteriaceae
enterobacteriaceae = [
    "Escherichia", "Klebsiella", "Enterobacter", "Citrobacter",
    "Serratia", "Proteus", "Morganella", "Salmonella", "Raoultella",
    "Pantoea"
]

def classify_microorganism(name):
    """
    Classifies a microorganism into defined label_map categories.
    The input is expected to be 'Genus_species' format.
    """
    # normalize
    specie = name.replace(".", "").strip()
    
    # --- Exact species-level mappings ---
    if specie == "Escherichia_coli":
        return "ECOLI"

    if specie == "Staphylococcus_aureus":
        return "SA"

    if specie == "Pseudomonas_aeruginosa":
        return "PSA"

    if specie == "Klebsiella_pneumoniae":
        return "KP"

    if specie == "Streptococcus_pneumoniae":
        return "SP"

    if specie.startswith("Enterococcus"):
        return "EC"

    genus = specie.split("_")[0]
    if genus in enterobacteriaceae:
        return "OEB"  # Enterobacteria

    else:
        return "NOEB"
