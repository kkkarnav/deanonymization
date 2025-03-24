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
from multiprocessing import Pool
import os

tqdm.pandas()
warnings.filterwarnings("ignore")


all_aux = []
all_aux_types = []


parser = argparse.ArgumentParser()
parser.add_argument("-fc", "--feature_count", help="The number of features available to the adversary", type=int)
parser.add_argument("-a", "--aux", help="Which variation of the auxiliary information function to use (mostinfo, random, public1, public2, public3)", type=str)
parser.add_argument("-s", "--sim", help="Which variation of the similarity function to use (perfect, twenty, presence)", type=str)
parser.add_argument("-r", "--robust", help="Whether to use the robust scoring function or not", type=bool)
args = parser.parse_args()


BASE_PATH = "./narayan"

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

def support_count(df, r):
    return ((df.iloc[r] != 0) & (df.iloc[r] != 0.0) & (df.iloc[r] != False)).sum()

def support_columns(df, r):
    return df[df.loc[r, df.loc[r].apply(lambda x: x != 0 and x != 0.0 and x != False)].index].columns

def support(df, r):
    return df[df.loc[r, df.loc[r].apply(lambda x: x != 0 and x != 0.0 and x != False)].index]

def support_values(df, r):
    return df[df.loc[r, df.loc[r].apply(lambda x: x != 0 and x != 0.0 and x != False)].index].iloc[r]

def support_count_c(df, c):
    return ((df.loc[:, c] != 0) & (df.loc[:, c] != 0.0) & (df.loc[:, c] != False)).sum()

def union_support_count(df, r1, r2):
    
    columns1 = support_columns(df, r1)
    columns2 = support_columns(df, r2)
    
    return len(list(set(columns1) | set(columns2)))

def union_support_columns(df, r1, r2):
    
    columns1 = support_columns(df, r1)
    columns2 = support_columns(df, r2)
    
    return list(set(columns1) | set(columns2))

# Sim(ri1, ri2) as defined by the paper
def similar(ri1, ri2):
    
    if type(ri1) is np.int64 or type(ri1) is int or type(ri1) is np.float64:
        ri1 = float(ri1)
    if type(ri2) is np.int64 or type(ri2) is int or type(ri2) is np.float64:
        ri2 = float(ri2)
    if (type(ri1) is bool and type(ri2) is np.bool_) or (type(ri1) is np.bool_ and type(ri2) is bool):
        ri1 = bool(ri1)
        ri2 = bool(ri2)
    
    if type(ri1) != type(ri2):
        return 0
    elif (type(ri1) is np.float64 and type(ri2) is np.float64) or (type(ri1) is float and type(ri2) is float):
        if args.sim == "perfect":
            diff = 1 if math.isclose(ri1, ri2, rel_tol=1e-9) else 0
        elif args.sim == "twenty":
            diff = 1 if abs(abs(ri2) - abs(ri1)) <= 0.2 * ri1 else 0
        elif args.sim == "presence":
            diff = 1 if math.isclose(ri1, ri2, rel_tol=1e-9) else 0
        return diff
    elif (type(ri1) is bool and type(ri2) is bool) or (type(ri1) is np.bool_ and type(ri2) is np.bool_):
        return 1 if ri1 == ri2 else 0
    elif type(ri1) is str and type(ri2) is str:
        return 1 if ri1 == ri2 else 0
    else:
        print(f"Uncaught, {type(ri1)}, {type(ri2)}")
        return 0

# Sim(r1, r2) as defined by the paper
def compute_similarity(df, r1, r2):
    
    similarity = pd.Series(union_support_columns(df, r1, r2)).apply(lambda column: similar(df.loc[r1, column], df.loc[r2, column])).sum()
    sim_score = similarity/union_support_count(df, r1, r2) if union_support_count(df, r1, r2) != 0 else 0
    return sim_score

# For the adversary's information, Aux(r). This randomly selects the columns that they know about
# We can consider a different method as well, based on some information measure, for eg some entropy measure or the number of rows that each column has a valid value for
def auxillary_columns(df, r, count):
    
    if args.aux == "mostinfo":
        supports = support_columns(df, r).to_series().apply(lambda c: support_count_c(df, c))
        aux = list(supports.sort_values().iloc[:min(count, len(supports))].index)
    elif args.aux == "random":
        aux = random.sample(list(support_columns(df, r)), min(count, support_count(df, r)))
    elif args.aux == "public1":
        population = list(set(list(df.columns[66:])).intersection(set(support_columns(df, r))))
        aux = random.sample(population, min(count, len(population)))
    elif args.aux == "public2":
        population = list(set(list(df.columns[66:493])).intersection(set(support_columns(df, r))))
        aux = random.sample(population, min(count, len(population)))
    elif args.aux == "public3":
        population = list(set(list(df.columns[362:])).intersection(set(support_columns(df, r))))
        aux = random.sample(population, min(count, len(population)))
    
    return aux

# Aux(r) as defined by the paper
def auxillary(df, r, count):
    
    # add the auxiliary columns select to the flat list all_aux
    all_aux.extend(auxillary_columns(df, r, count))
    all_aux_types.extend([type(df.loc[r, col]) for col in auxillary_columns(df, r, count)])
    
    aux = df.loc[r, auxillary_columns(df, r, count)]
    return aux

# Score(aux, r') as defined by the paper
def score(df, aux, r0):
    
    scores = pd.Series(aux.index).apply(lambda column: similar(aux.loc[column], df.loc[r0, column]))
    return scores.sum()/len(aux.index) if len(aux.index) != 0 else 0

# Robust Score(aux, r') as defined by the paper
def score_robust(df, aux, r0):
    
    scores = pd.Series(aux.index).apply(lambda column: (math.e**similar(aux.loc[column], df.loc[r0, column]))/max(1, math.log(support_count_c(df, column))))
    return scores.sum()/len(aux.index) if len(aux.index) != 0 else 0

# Best guess as defined by the paper
def best_guess(candidates, eccentricity):
    sorted_candidates = candidates.sort_values(ascending=False)
    heuristic = (sorted_candidates.index[0] - sorted_candidates.index[1])/candidates.std()
    return sorted_candidates.index[0] if heuristic >= eccentricity else None

# Matching criterion as defined by the paper
def match(df, aux, alpha):
    
    candidates = pd.Series(df.index).apply(lambda row: score(df, aux, row))
    candidates = candidates[candidates >= alpha]
    if candidates.empty:
        return pd.Series({"-1": 0})
    return candidates

# Robust matching criterion as defined by the paper
def match_robust(df, aux, alpha):
    
    candidates = pd.Series(df.index).apply(lambda row: score_robust(df, aux, row))
    if candidates.empty:
        return pd.Series({"-1": 0})
    elif candidates.shape[0] > 1:
        return best_guess(candidates, 1.5)
    return candidates

# Scoreboard() as defined by the paper
def scoreboard(df, count):
    
    aux_series = pd.Series(df.index).apply(lambda r1: auxillary(df, r1, count))
    scores = pd.DataFrame(index=df.index, columns=df.index)
    
    for r1 in tqdm(list(df.index)):
        scores.loc[r1, :] = pd.Series(df.index).apply(lambda r2: score(df, aux_series.loc[r1, ~aux_series.loc[r1].isna()], r2))
    
    return scores


def compute_row(r1, df, aux_series):
    return pd.Series(df.index).apply(lambda r2: score(df, aux_series.loc[r1, ~aux_series.loc[r1].isna()], r2))

def scoreboard_parallel(df, count):
    aux_series = pd.Series(df.index).apply(lambda r1: auxillary(df, r1, count))
    scores = pd.DataFrame(index=df.index, columns=df.index)
    
    with Pool() as pool:
        results = list(tqdm(pool.starmap(compute_row, [(r1, df, aux_series) for r1 in df.index]), total=len(df.index)))
    
    for r1, result in tqdm(zip(df.index, results)):
        scores.loc[r1] = result
    
    return scores


def scoreboard_robust(df, count):
    
    aux_series = pd.Series(df.index).apply(lambda r1: auxillary(df, r1, count))
    scores = pd.DataFrame(index=df.index, columns=df.index)
    
    for r1 in tqdm(list(df.index)):
        scores.loc[r1, :] = pd.Series(df.index).apply(lambda r2: score_robust(df, aux_series.loc[r1, ~aux_series.loc[r1].isna()], r2))
            
    return scores


def compute_row_robust(r1, df, aux_series):
    return pd.Series(df.index).apply(lambda r2: score_robust(df, aux_series.loc[r1, ~aux_series.loc[r1].isna()], r2))


def scoreboard_robust_parallel(df, count):
    aux_series = pd.Series(df.index).apply(lambda r1: auxillary(df, r1, count))
    scores = pd.DataFrame(index=df.index, columns=df.index)
    
    with Pool() as pool:
        results = list(tqdm(pool.starmap(compute_row_robust, [(r1, df, aux_series) for r1 in df.index]), total=len(df.index)))
    
    for r1, result in zip(df.index, results):
        scores.loc[r1] = result
    
    return scores


df3 = data.loc[:500, [col for col in data.columns]]

# generate the presence-only df
shares = data.columns.to_series().apply(lambda x: int((list(data.loc[:, x].value_counts())[0])/len(data)*100))
roundshares = shares.apply(lambda x: True if x > 50 else False)

data = data.iloc[:1000]
df4 = data.progress_apply(lambda col: col.apply(lambda x: x != col.mode()[0]) if roundshares.get(col.name, False) else col)
df4 = df4.select_dtypes(include=[bool])


whether_robust = "robust" if args.robust else "regular"
output_dir = f"{BASE_PATH}/results/r_{args.aux}_{args.sim}_{whether_robust}"
os.makedirs(output_dir, exist_ok=True)

if whether_robust == "robust":
    if args.sim == "presence":
        results = scoreboard_robust(df4, args.feature_count)
    else:
        results = scoreboard_robust(df3, args.feature_count)
else:
    if args.sim == "presence":
        results = scoreboard(df4, args.feature_count)
    else:
        results = scoreboard(df3, args.feature_count)

# results.to_csv(f"{output_dir}/results_{args.feature_count}features.csv", index=False)
print(f"run for args: {args.aux}, {args.sim}, {args.robust}, {args.feature_count}")
all_aux = pd.Series(all_aux)
all_aux_types = pd.Series(all_aux_types)
print(all_aux.value_counts())
print(all_aux_types.value_counts())
print(all_aux_types.value_counts(normalize=True))
all_aux.to_csv(f"./aux_{args.feature_count}features.csv", index=False)
all_aux_types.to_csv(f"./aux_types_{args.feature_count}features.csv", index=False)
