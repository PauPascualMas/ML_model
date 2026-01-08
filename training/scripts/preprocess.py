#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from pandas_profiling import ProfileReport
from utils import classify_cancer, classify_microorganism

import warnings
warnings.filterwarnings('ignore')

# ----------------------
# Paths
# ----------------------
DATA_DIR = "../data"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)
bacthecom_db=f"{DATA_DIR}/db_bacthecom_v5.db"

# ----------------------
# Connect to DB
# ----------------------
conn = sqlite3.connect(bacthecom_db)

# ----------------------
# Load tables
# ----------------------
tbl_pacientes = pd.read_sql_query("SELECT * FROM paciente", con=conn)
tbl_factores_bmr = pd.read_sql_query("SELECT * FROM factores_riesgo_infeccion_bmr", con=conn)
tbl_episodios = pd.read_sql_query("SELECT * FROM episodio_ingreso", con=conn)
tbl_comorbilidades = pd.read_sql_query("SELECT * FROM comorbilidad", con=conn)
tbl_signos = pd.read_sql_query("SELECT * FROM signos_sintomas", con=conn)
tbl_microorganismo = pd.read_sql_query("SELECT * FROM episodio_infeccion", con=conn)
tbl_antibiograma = pd.read_sql_query("SELECT * FROM antibiograma", con=conn)
tbl_tto = pd.read_sql_query("SELECT * FROM tto_antimicrobiano", con=conn)

# ----------------------
# Patients preprocessing
# ----------------------
tbl_pacientes['fecha_nacimiento'] = pd.to_datetime(tbl_pacientes['fecha_nacimiento'], errors='coerce').dt.date
# Recode sexo
tbl_pacientes['sexo'] = tbl_pacientes['sexo'].map({'Hombre': 0, 'Mujer': 1})

# ----------------------
# Episodes preprocessing
# ----------------------
tbl_episodios['fecha_ingreso'] = pd.to_datetime(tbl_episodios['fecha_ingreso'], errors='coerce').dt.date
tbl_episodios['IRAs_nosocomial'] = tbl_episodios['IRAs_nosocomial'].map({'No': 0, 'Si': 1})
tbl_episodios.drop(columns=['codigo_postal'], inplace=True)

# Rename organo_aparato
infecname_to_infeccode = {
    'via_urinaria_superior': 'Infección de la vía urinaria superior',
    'fiebre_sin_foco': 'Fiebre sin foco',
    'cateter_vascular': 'Infeccion de cateter vascular',
    'vias_biliares': 'Infeccion vias biliares',
    'intraabdominal': 'Infeccion intraabdominal',
    'tracto_respiratorio_inferior': 'Infeccion tracto respiratorio inferior',
    'piel': 'Infeccion de piel y partes blandas',
    'cardiovascular': 'Infeccion cardiovascular',
    'osteoarticular': 'Infeccion osteoarticular',
    'etiologia_incierta': 'Otros/Infeccion de etiologia incierta',
    'snc': 'Infeccion del SNC',
    'genital': 'Infeccion genital'
}

tbl_episodios['organo_aparato'] = tbl_episodios['organo_aparato'].map({v: k for k, v in infecname_to_infeccode.items()})
tbl_episodios_recoded = pd.get_dummies(tbl_episodios, columns=['organo_aparato'], prefix='infec', dtype=int)

# ----------------------
# Comorbidities preprocessing
# ----------------------
tbl_comorbilidades['fecha_ingreso'] = pd.to_datetime(tbl_comorbilidades['fecha_ingreso'], errors='coerce').dt.date

# Cancer classification
tbl_comorbilidades_ = tbl_comorbilidades.copy()
tbl_comorbilidades_['tipo_cancer_list'] = tbl_comorbilidades_['tipo_cancer'].dropna().apply(lambda x: x.split(","))
tbl_comorbilidades_exploded = tbl_comorbilidades_.explode('tipo_cancer_list')
tbl_comorbilidades_exploded['tipo_cancer_list'] = tbl_comorbilidades_exploded['tipo_cancer_list'].astype(str).str.strip().str.split('.').str[0]
tbl_comorbilidades_exploded['cancer_class'] = tbl_comorbilidades_exploded['tipo_cancer_list'].map(classify_cancer)
dummies = pd.get_dummies(tbl_comorbilidades_exploded['cancer_class'], prefix='has_cancer', dtype=int).assign(record_id=tbl_comorbilidades_exploded['record_id']).groupby('record_id').max()

# Merge
tbl_comorbilidades_recoded = tbl_comorbilidades.merge(dummies, on='record_id', how='left')
for col in ['puntaje_child_pugh','hepatopatia_ligera','hepatopatia_moderada_o_grave']:
    tbl_comorbilidades_recoded[col] = np.where(tbl_comorbilidades_recoded[col].isna(), 0,1)

# Drop unneeded columns
cols_drop = ['tipo_cancer','tipo_hepatopatia','causa_inmunosupresion','fecha_TOS','fecha_TPH','clasificacion_quemadura']
tbl_comorbilidades_recoded.drop(columns=cols_drop, inplace=True, errors='ignore')

# ----------------------
# Signs preprocessing
# ----------------------
tbl_signos['fecha_ingreso'] = pd.to_datetime(tbl_signos['fecha_ingreso'], errors='coerce').dt.date
tbl_signos.drop(columns=['situacion_funcional_basal'], inplace=True)
tbl_signos['foco'] = tbl_signos['foco'].replace({'piel y partes blandas': 'piel', 'cateter venoso':'cateter'})
tbl_signos_recoded = pd.get_dummies(tbl_signos, columns=['foco'], prefix='foco', dtype=int)
tbl_signos_recoded['somnolencia_estupor_coma'] = np.where(tbl_signos_recoded['somnolencia_estupor_coma'].isna(),0,1)
tbl_signos_recoded['charlson_index_class'] = pd.cut(tbl_signos['indice_de_charlson'],bins=[-1,3,6,float('inf')],labels=['low','medium','high'])
tbl_signos_recoded = pd.get_dummies(tbl_signos_recoded, columns=['charlson_index_class'], prefix='charlson_index', dtype=int)

# Vital signs flags
tbl_signos_recoded['hipotermia_hipertermia'] = np.where(tbl_signos_recoded['temperatura']>=38,2,np.where(tbl_signos_recoded['temperatura']>=36,0,1))
tbl_signos_recoded['hipotermia_hipertermia'] = tbl_signos_recoded['hipotermia_hipertermia'].where(tbl_signos['temperatura'].notna())
tbl_signos_recoded['hipotension'] = np.where((tbl_signos['tension_arterial_sist']<=90)&(tbl_signos['tension_arterial_diast']<=60),1,0)
tbl_signos_recoded['hipertension'] = np.where((tbl_signos['tension_arterial_sist']>=140)&(tbl_signos['tension_arterial_diast']>=90),1,0)
tbl_signos_recoded['taquipnea'] = np.where(tbl_signos['frecuencia_respiratoria'].isna(),np.nan,np.where(tbl_signos['frecuencia_respiratoria']>20,1,0))
tbl_signos_recoded['taquicardia'] = np.where(tbl_signos['frec_cardiaca'].isna(),np.nan,np.where(tbl_signos['frec_cardiaca']>90,1,0))
tbl_signos_recoded['hipoxemia'] = np.where(tbl_signos['saturacion_pO2'].isna(),np.nan,np.where(tbl_signos['saturacion_pO2']<0.90,1,0))
cols_drop_signos = ['indice_de_charlson','escala_karnofsky','temperatura','tension_arterial_sist','tension_arterial_diast','frec_cardiaca','saturacion_pO2']
tbl_signos_recoded.drop(columns=cols_drop_signos, inplace=True)

# ----------------------
# Microorganisms preprocessing
# ----------------------
tbl_microorganismo = tbl_microorganismo.sort_values(['record_id','fecha_ingreso','fecha_cultivo'])
tbl_microorganismo.drop(columns=['area_hosp','id_cultivo'], inplace=True)
tbl_microorganismo['fecha_cultivo'] = pd.to_datetime(tbl_microorganismo['fecha_cultivo']).dt.date
tbl_microorganismo['fecha_ingreso'] = pd.to_datetime(tbl_microorganismo['fecha_ingreso']).dt.date

PRE_ADMISSION_DAYS=2
# Filter blood cultures
valid = tbl_microorganismo[(tbl_microorganismo['especimen']=='Sangre')&(tbl_microorganismo['fecha_cultivo']>=tbl_microorganismo['fecha_ingreso'] - pd.Timedelta(days=PRE_ADMISSION_DAYS))].copy()
first_hemo_dates = valid.groupby(['record_id','fecha_ingreso'])['fecha_cultivo'].min().reset_index().rename(columns={'fecha_cultivo':'first_hemo_date'})
valid = valid.merge(first_hemo_dates, on=['record_id','fecha_ingreso'])
valid['hemocultivo_principal'] = (valid['fecha_cultivo']==valid['first_hemo_date']).astype(int)
tbl_microorganismo = tbl_microorganismo.merge(valid[['record_id','fecha_ingreso','episode_id','hemocultivo_principal']], on=['record_id','fecha_ingreso','episode_id'], how='left')
tbl_microorganismo['hemocultivo_principal'] = tbl_microorganismo['hemocultivo_principal'].fillna(0).astype(int)
first_hemo_map = first_hemo_dates.set_index(['record_id','fecha_ingreso'])['first_hemo_date']
tbl_microorganismo['fecha_principal_hemo'] = tbl_microorganismo.set_index(['record_id','fecha_ingreso']).index.map(first_hemo_map)
tbl_microorganismo['episodio_previo_si_no'] = np.where((tbl_microorganismo['fecha_principal_hemo'].notna())&(tbl_microorganismo['fecha_cultivo']<tbl_microorganismo['fecha_principal_hemo']),1,0)
tbl_microorganismo.drop(columns=['fecha_principal_hemo'], inplace=True)

# Classify microorganisms
tbl_microorganismo['microorganismo_recoded'] = tbl_microorganismo['microorganismo'].fillna('').apply(lambda x: '_'.join(x.split(' ')[:2]))
tbl_microorganismo['microorganismo_group'] = tbl_microorganismo['microorganismo_recoded'].map(classify_microorganism)

# One-hot main vs previous
micro_main_cols = pd.get_dummies(tbl_microorganismo.loc[tbl_microorganismo['episodio_previo_si_no']==0,'microorganismo_group'], prefix='microorganismo_main', dtype=int)
micro_prev_cols = pd.get_dummies(tbl_microorganismo.loc[tbl_microorganismo['episodio_previo_si_no']==1,'microorganismo_group'], prefix='microorganismo_prev', dtype=int)
tbl_microorganismo = tbl_microorganismo.join(micro_main_cols).join(micro_prev_cols)
tbl_microorganismo['BMR_main_si_no'] = np.where((tbl_microorganismo['episodio_previo_si_no']==0)&(tbl_microorganismo['fenotipo_resistencia'].notna()),1,0)
tbl_microorganismo['BMR_prev_si_no'] = np.where((tbl_microorganismo['episodio_previo_si_no']==1)&(tbl_microorganismo['fenotipo_resistencia'].notna()),1,0)
tbl_microorganismo_recoded = tbl_microorganismo.drop(columns=['microorganismo','fenotipo_resistencia'])
tbl_microorganismo_recoded.fillna(0, inplace=True)

# Collapse per record_id
micro_main_cols = [c for c in tbl_microorganismo_recoded.columns if c.startswith('microorganismo_main_')]
micro_prev_cols = [c for c in tbl_microorganismo_recoded.columns if c.startswith('microorganismo_prev_')]
df_main = tbl_microorganismo_recoded[(tbl_microorganismo_recoded['episodio_previo_si_no']==1)|(tbl_microorganismo_recoded['hemocultivo_principal']==1)].copy()
df_max_main = df_main.groupby(['record_id','fecha_ingreso'])[micro_main_cols].max()
df_max_prev = df_main.groupby(['record_id','fecha_ingreso'])[micro_prev_cols].max()
df_first = df_main.sort_values(['record_id','fecha_ingreso','fecha_cultivo']).groupby(['record_id','fecha_ingreso']).first()
df_first[micro_main_cols] = df_max_main
df_first[micro_prev_cols] = df_max_prev
tbl_microorganismo_collapsed = df_first.reset_index()
tbl_microorganismo_collapsed.drop(columns=['especimen','microorganismo_recoded','fecha_cultivo','microorganismo_group'], inplace=True, errors='ignore')

# ----------------------
# Merge all tables
# ----------------------
df_merged = df_merged = pd.merge(tbl_pacientes, tbl_episodios_recoded, on='record_id', how='left')
fecha_ingreso_dt = pd.to_datetime(df_merged['fecha_ingreso'], errors='coerce')
fecha_nac_dt = pd.to_datetime(df_merged['fecha_nacimiento'], errors='coerce')
df_merged['edad'] = ((fecha_ingreso_dt - fecha_nac_dt).dt.days // 365)
df_merged.drop(columns=['fecha_nacimiento'], inplace=True)

for table in [tbl_comorbilidades_recoded, tbl_microorganismo_collapsed, tbl_signos_recoded, tbl_factores_bmr]:
    df_merged = pd.merge(df_merged, table, on=['record_id','fecha_ingreso'], how='left')

df_merged = pd.merge(df_merged, tbl_antibiograma.groupby('episode_id').agg(list).reset_index(), on='episode_id', how='left')

# Drop known mismatches
mismatches_record = [60,427,1341,1741,2356]
df_merged = df_merged[~df_merged['record_id'].isin(mismatches_record)]

# ----------------------
# Save preprocessed df
# ----------------------
df_merged.to_csv(f"{DATA_DIR}/preprocessed_df.csv", index=False)
df_merged.to_pickle(f"{DATA_DIR}/preprocessed_df.pkl")

# ----------------------
# Generate profile report
# ----------------------
profile = ProfileReport(df_merged, title="Bacthecom Data Report", explorative=True)
profile.to_file(os.path.join(RESULTS_DIR, "bacthecom_report.html"))