import pandas as pd
grupos_cancer_icd10 = {
    "comportamiento": {
        "maligno":        range(0, 100),   # Se aplica solo a códigos Cxx
        "in_situ":        range(0, 10),    # D00–D09
        "benigno":        range(10, 37),   # D10–D36
        "incierto":       range(37, 49),   # D37–D48
    },

    "localizacion_C": {  # Solo códigos que empiezan en C
        "cavidad_oral_faringe":   range(0, 15),   # C00–C14
        "digestivo":              range(15, 27),  # C15–C26
        "respiratorio":           range(30, 40),  # C30–C39
        "hueso_cartilago":        range(40, 42),  # C40–C41
        "piel_melanoma":          range(43, 45),  # C43–C44
        "mama":                   range(50, 51),  # C50
        "genital_femenino":       range(51, 59),  # C51–C58
        "genital_masculino":      range(60, 64),  # C60–C63
        "renal_urinario":         range(64, 69),  # C64–C68
        "ojo_snc":                range(69, 73),  # C69–C72
        "endocrino":              range(73, 76),  # C73–C75
        "mal_definido_metastasis":range(76, 81),  # C76–C80
    },

    "localizacion_D": {  # Solo códigos que empiezan en D
        "in_situ":                range(0, 10),   # D00–D09
        "benigno_boca_faringe":   range(10, 12),  # D10–D11
        "benigno_digestivo":      range(12, 20),  # D12–D19
        "benigno_respiratorio":   range(20, 23),  # D20–D22
        "benigno_otros":          range(23, 37),  # D23–D36
        "incierto":               range(37, 49),  # D37–D48
    }
}
def recode_tipo_cancer(icd10_code):
    if pd.isna(icd10_code):
        return None

    code = icd10_code.strip().upper()
    if not code or len(code) < 3:
        return None

    letter = code[0]
    try:
        number = int(code[1:3])
    except ValueError:
        return None

    if letter == 'C':
        for category, ranges in grupos_cancer_icd10['localizacion_C'].items():
            if number in ranges:
                return category
        if number in grupos_cancer_icd10['comportamiento']['maligno']:
            return 'maligno_no_especificado'
    elif letter == 'D':
        for category, ranges in grupos_cancer_icd10['localizacion_D'].items():
            if number in ranges:
                return category
        for category, ranges in grupos_cancer_icd10['comportamiento'].items():
            if number in ranges:
                return category

    return 'otro_o_no_especificado'