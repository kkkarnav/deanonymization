import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import argparse

tqdm.pandas()
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("-rl", "--relative", help="The two relatives to match", type=str, default="s2f", choices=["s2f", "s2m", "f2gf", "m2gm", "s2s", "f2f", "m2m", "gf2gf", "gm2gm"])
args = parser.parse_args()


BASE_PATH = "."

data = pd.read_csv(f"{BASE_PATH}/max_data/max_dataset.csv")
data = data.fillna("Nil")
data.replace("0", "Nil", inplace=True)
data.replace("Nil", 0, inplace=True)
data.replace("Negative", 0, inplace=True)
data.replace("Normal", 0, inplace=True)
data.replace("Absent", 0, inplace=True)
data.replace("Yes", True, inplace=True)
data.replace("No", False, inplace=True)
data.replace("yes", True, inplace=True)
data.replace("no", False, inplace=True)
data.replace("True", True, inplace=True)
data.replace("False", False, inplace=True)
data = data.select_dtypes(include=[int, float, bool, object])
data = data.drop(columns=[data.columns[0], data.columns[1]])
data["PDOB.dob"] = data["PDOB.dob"].apply(lambda x: int(x.split("-")[-1]))

# print the type of each column in the df
types = [data[col].dtype for col in data.columns]

# print value counts of types
print("Value counts of types:")
print(pd.Series(types).value_counts())

desc = pd.read_excel(f"{BASE_PATH}/max_data/003_data-description-OMOP_coded-31Aug23--PPRK.xlsx").iloc[2:, :]
desc["lifestyle"] = desc["field_information"].apply(lambda x: x in ["Alcohol History", "Physical Activity", "Social and Personal History ", "Allergies ", "Travel History", "Personal", "Food habits", "Sleep"])
desc["medical"] = desc["field_information"].apply(lambda x: x in ["Current Medications", "Past Medical History", "FAMILY HISTORY: Father", "Siblings/Others", "Grand Mother", "Grand Father", "Aunty", "Mother", "Uncle", "Investigation Comments/ Reports ", "General Examination", "Past Surgical History", "Systemic Examination", "Allergies ", "Anthropometric Data"])
desc["measure"] = desc["field_information"].apply(lambda x: x in ["Blood Test Report ", "Current Medications", "Past Medical History", "Anthropometric Data", "Vitals", "Radiographs/ Spirometry", "Personal"])
desc["family"] = desc["field_information"].apply(lambda x: x in ["FAMILY HISTORY: Father", "Siblings/Others", "Grand Mother", "Grand Father", "Aunty", "Mother", "Uncle"])
desc["cat"] = desc.apply(lambda x: "Lifestyle" if x["lifestyle"] else "Medical" if x["medical"] else "Measure" if x["measure"] else "Nothing", axis=1)
desc["type"] = desc.apply(lambda x: "Medical Data" if x["field_information"] in ["Blood Test Report", "Vitals", "Anthropometric Data", "Radiographs/ Spirometry", "Investigation Comments/ Reports ", "General Examination", "Systemic Examination"] else "Survey Response", axis=1)

self_rows = desc[(desc["field_information"] == "Past Medical History") & (~desc["php_code_header"].str.contains("yearofend|duration|HPWO|HPWS|HPXS|HPXO|HPPO.others2|HPPS"))]["php_code_header"]
father_rows = desc[(desc["family"] == True) & (desc["field_information"] == "FAMILY HISTORY: Father") & (~desc["php_code_header"].str.contains("yearofend|duration"))]["php_code_header"]
mother_rows = desc[(desc["family"] == True) & (desc["field_information"] == "Mother") & (~desc["php_code_header"].str.contains("yearofend|duration"))]["php_code_header"]
grandfather_rows = desc[(desc["family"] == True) & (desc["field_information"] == "Grand Father") & (~desc["php_code_header"].str.contains("yearofend|duration"))]["php_code_header"]
grandmother_rows = desc[(desc["family"] == True) & (desc["field_information"] == "Grand Mother") & (~desc["php_code_header"].str.contains("yearofend|duration"))]["php_code_header"]
sibling_rows = desc[(desc["family"] == True) & (desc["field_information"] == "Siblings/Others") & (~desc["php_code_header"].str.contains("yearofend|duration"))]["php_code_header"]
self = data.loc[:, self_rows]
father = data.loc[:, father_rows]
mother = data.loc[:, mother_rows]
grandfather = data.loc[:, grandfather_rows]
grandmother = data.loc[:, grandmother_rows]
sibling = data.loc[:, sibling_rows]

print(args.relative)

if args.relative == "s2f":
    self_against_father = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, self_rows].replace(0, False).any() & 
                    (data.loc[x, self_rows].values == data.loc[y, father_rows].values).all() & 
                    (data.loc[y, "PGDR.gender"] == "Male") & 
                    ((data.loc[y, "PDOB.dob"] - data.loc[x, "PDOB.dob"]) >= 18)
                )
            )
        )
    )
    self_against_father.to_csv(f"{BASE_PATH}/results/nb/self_against_father.csv", index=False)
elif args.relative == "s2m":
    self_against_mother = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, self_rows].replace(0, False).any() & 
                    (data.loc[x, self_rows].values == data.loc[y, mother_rows].values).all() & 
                    (data.loc[y, "PGDR.gender"] == "Female") & 
                    ((data.loc[y, "PDOB.dob"] - data.loc[x, "PDOB.dob"]) >= 18)
                )
            )
        )
    )
    self_against_mother.to_csv(f"{BASE_PATH}/results/nb/self_against_mother.csv", index=False)
elif args.relative == "f2gf":
    father_against_grandfather = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, father_rows].replace(0, False).any() & 
                    (data.loc[x, father_rows].values == data.loc[y, grandfather_rows].values).all() & 
                    (data.loc[y, "PGDR.gender"] == "Male") & 
                    ((data.loc[y, "PDOB.dob"] - data.loc[x, "PDOB.dob"]) >= 36)
                )
            )
        )
    )
    father_against_grandfather.to_csv(f"{BASE_PATH}/results/nb/father_against_grandfather.csv", index=False)
elif args.relative == "m2gm":
    mother_against_grandmother = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, mother_rows].replace(0, False).any() & 
                    (data.loc[x, mother_rows].values == data.loc[y, grandmother_rows].values).all() & 
                    (data.loc[y, "PGDR.gender"] == "Female") & 
                    ((data.loc[y, "PDOB.dob"] - data.loc[x, "PDOB.dob"]) >= 36)
                )
            )
        )
    )
    mother_against_grandmother.to_csv(f"{BASE_PATH}/results/nb/mother_against_grandmother.csv", index=False)
elif args.relative == "s2s":
    self_against_sibling = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, self_rows].replace(0, False).any() & 
                    (data.loc[x, self_rows].values == data.loc[y, sibling_rows].values).all()
                )
            )
        )
    )
    self_against_sibling.to_csv(f"{BASE_PATH}/results/nb/self_against_sibling.csv", index=False)
elif args.relative == "f2f":
    father_against_father = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, father_rows].replace(0, False).any() & 
                    (data.loc[x, father_rows].values == data.loc[y, father_rows].values).all()
                )
            )
        )
    )
    father_against_father.to_csv(f"{BASE_PATH}/results/nb/father_against_father.csv", index=False)
elif args.relative == "m2m":
    mother_against_mother = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, mother_rows].replace(0, False).any() & 
                    (data.loc[x, mother_rows].values == data.loc[y, mother_rows].values).all()
                )
            )
        )
    )
    mother_against_mother.to_csv(f"{BASE_PATH}/results/nb/mother_against_mother.csv", index=False)
elif args.relative == "gf2gf":
    gfather_against_gfather = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, grandfather_rows].replace(0, False).any() & 
                    (data.loc[x, grandfather_rows].values == data.loc[y, grandfather_rows].values).all()
                )
            )
        )
    )
    gfather_against_gfather.to_csv(f"{BASE_PATH}/results/nb/gfather_against_gfather.csv", index=False)
elif args.relative == "gm2gm":
    gmother_against_gmother = (
    data.index.to_series().progress_apply(
            lambda y: data.index.to_series().apply(
                lambda x: (
                    data.loc[x, grandmother_rows].replace(0, False).any() & 
                    (data.loc[x, grandmother_rows].values == data.loc[y, grandmother_rows].values).all()
                )
            )
        )
    )
    gmother_against_gmother.to_csv(f"{BASE_PATH}/results/nb/gmother_against_gmother.csv", index=False)
