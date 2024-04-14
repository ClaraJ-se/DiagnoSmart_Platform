import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import get_path

def get_subsets(df: pd.DataFrame):
   
   df_men = df[df.gender == 'M']
   df_women = df[df.gender == 'F']
   df_infarctus_pos = df[df.infarctus == 1]
   df_infarctus_neg = df[df.infarctus == 0]
   df_infarctus_pos_women = df_infarctus_pos[df_infarctus_pos.gender == 'F']
   df_infarctus_neg_women = df_infarctus_neg[df_infarctus_neg.gender == 'F']
   df_infarctus_pos_men = df_infarctus_pos[df_infarctus_pos.gender == 'M']
   df_infarctus_neg_men = df_infarctus_neg[df_infarctus_neg.gender == 'M']

   
   print(f"Women: Number Acute myocardial infarction: {df_infarctus_pos_women['stay_id'].nunique()}")
   print(f"Women: Total number of stays: {df_women['stay_id'].nunique()}")
   print(f"Women: Rate of Acute myocardial infarction: {df_infarctus_pos_women['stay_id'].nunique()/df_women['stay_id'].nunique()}")  

   print(f"Men: Number Acute myocardial infarction: {df_infarctus_pos_men['stay_id'].nunique()}")
   print(f"Men: Total number of stays: {df_men['stay_id'].nunique()}")
   print(f"Men: Rate of Acute myocardial infarction: {df_infarctus_pos_men['stay_id'].nunique()/df_men['stay_id'].nunique()}")  

   print(f"Infarctus: Rate of women: {df_infarctus_pos_women['stay_id'].nunique()/df_infarctus_pos['stay_id'].nunique()}")   

   return df_men, df_women, df_infarctus_pos, df_infarctus_neg, df_infarctus_pos_women, df_infarctus_neg_women, df_infarctus_pos_men, df_infarctus_neg_men



if __name__ == "__main__":
   paths = get_path()   

   parameters = pd.read_csv(os.path.join(paths['wrangling'],'parameters.csv'))
   
   infarctus = pd.read_csv(os.path.join(paths['wrangling'],'infarctus.csv'))
   
   edstays = pd.read_csv(os.path.join(paths['raw_ed'],'edstays.csv'))  

   #! before to select patients
   
   list_potential_temp = pd.merge(edstays, infarctus[['stay_id','infarctus']].drop_duplicates(),how='left')   

   output_subset_1 = get_subsets(list_potential_temp)
   df_men, df_women, df_infarctus_pos, df_infarctus_neg, df_infarctus_pos_women, df_infarctus_neg_women, df_infarctus_pos_men, df_infarctus_neg_men = output_subset_1

   #? Women: Number Acute myocardial infarction: 944
   #? Women: Total number of stays: 229898
   #? Women: Rate of Acute myocardial infarction: 0.004106168822695282 
       
   #? Men: Number Acute myocardial infarction: 1347
   #? Men: Total number of stays: 195189
   #? Men: Rate of Acute myocardial infarction: 0.006901003642623304   
       
   #? Infarctus: Rate of women: 0.4120471409864688   
       
   #! select patients to obtain around 50% infarctus. Among neg, 40% of women
   
   #! Seconde version: 50/50 but need to remove some men infarctus  
       
   list_potential_patients = parameters['stay_id'].drop_duplicates()   
   list_potential = pd.merge(list_potential_patients, edstays,how='left')
   
   list_potential = pd.merge(list_potential, infarctus[['stay_id','infarctus']].drop_duplicates(),how='left') 

   print(f"Number Acute myocardial infarction: {list_potential[list_potential['infarctus']==1]['stay_id'].nunique()}")
   print(f"Total number of stays: {list_potential['stay_id'].nunique()}")
   print(f"Rate of Acute myocardial infarction: {list_potential[list_potential['infarctus']==1]['stay_id'].nunique()/list_potential['stay_id'].nunique()}")   


   output_subset_2 = get_subsets(list_potential)
   df_men, df_women, df_infarctus_pos, df_infarctus_neg, df_infarctus_pos_women, df_infarctus_neg_women, df_infarctus_pos_men, df_infarctus_neg_men = output_subset_2

   #? Women: Number Acute myocardial infarction: 944   
   #? Women: Total number of stays: 229898
   #? Women: Rate of Acute myocardial infarction: 0.0041   
       
   #? Men: Number Acute myocardial infarction: 1347
   #? Men: Total number of stays: 195189
   #? Men: Rate of Acute myocardial infarction: 0.0069
   
   #? Infarctus: Rate of women: 0.4120471409864688

   #! remove when the patient don't have symptom at all
   list_symptoms = [ 'pain','jaundice','hyperglycemia', 'dehydration', 'Hematemesis', 'distention', 'nausea',
       'swelling', 'tachycardia', 'bleed', 'fatigue', 'fever', 'cough', 'itch',
       'paralysis', 'diarrhea', 'dizzy', 'hemorroids', 'neurologic',
       'lump', 'numbness', 'seizure', 'migraine', 'sore', 'smelling urine',
       'hearing loss', 'rash_redness', 'hypoglycemia', 'dyspnea', 'anemia',
       'throat foreign body sensation', 'constipation', 'dysuria', 'anxiety',
       'hematuria', 'pain_back', 'pain_neck',
       'pain_chest', 'pain_joint', 'pain_abdominal', 'pain_head',
       'pain_urinary track', 'paralysis_face', 'paralysis_arm',
       'cramps_abdominal', 'pain_arm_left']
    # remove patients with no symptoms
   df_pat = parameters.copy()
   df_pat['number_symptoms'] = df_pat[list_symptoms].sum(axis=1)
   df_pat = df_pat[df_pat.number_symptoms >=1]

   print(f"Total number of stays with symptoms: {df_pat['stay_id'].nunique()}")

   #! remove when missing data in vitals
   list_vital = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
   
   df_pat.dropna(how='any',inplace=True)
   
   print(f"Total number of stays with symptoms: {df_pat['stay_id'].nunique()}")

   #! List of pre selected patients
   df_accepted = pd.merge(df_pat,list_potential,how='left')
     
   output_subset_3 = get_subsets(df_accepted)
   df_men, df_women, df_infarctus_pos, df_infarctus_neg, df_infarctus_pos_women, df_infarctus_neg_women, df_infarctus_pos_men, df_infarctus_neg_men = output_subset_3

   #! select the same number of patients for men and women than for infarctus positive
   # select 944 women without infarctus and men 1347 without
   selected_pos_men = df_infarctus_pos_men.sample(df_infarctus_pos_women['stay_id'].nunique())
   selected_neg_women = df_infarctus_neg_women.sample(df_infarctus_pos_women['stay_id'].nunique())
   selected_neg_men = df_infarctus_neg_men.sample(df_infarctus_pos_women['stay_id'].nunique() )

   selected_patients = pd.concat([selected_pos_men,selected_neg_women,selected_neg_men,df_infarctus_pos_women])
   selected_patients = selected_patients.sort_values(by='stay_id')

   selected_patients.to_csv(os.path.join(paths['selection'],'patients.csv'))

#? Women: Number Acute myocardial infarction: 362
#? Women: Total number of stays: 723
#? Women: Rate of Acute myocardial infarction: 0.5006915629322268
#? Men: Number Acute myocardial infarction: 362
#? Men: Total number of stays: 723
#? Men: Rate of Acute myocardial infarction: 0.5006915629322268
#? Infarctus: Rate of women: 0.5