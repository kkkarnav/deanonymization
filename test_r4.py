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

BASE_PATH = "./"

df = pd.read_csv(f"{BASE_PATH}/max_data/max_dataset.csv")
df = df.fillna("Nil")
df.replace("0", "Nil", inplace=True)
df.replace("Nil", 0, inplace=True)
df.replace("Negative", 0, inplace=True)
df.replace("Normal", 0, inplace=True)
df.replace("Absent", 0, inplace=True)
df.replace("Yes", True, inplace=True)
df.replace("No", False, inplace=True)
df.replace("yes", True, inplace=True)
df.replace("no", False, inplace=True)
df.replace("True", True, inplace=True)
df.replace("False", False, inplace=True)
df = df.select_dtypes(include=[int, float, bool, object])
df = df.drop(columns=[df.columns[0], df.columns[1]])

# print the type of each column in the df
types = [df[col].dtype for col in df.columns]

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
    
    if type(ri1) != type(ri2):
        return 0
    elif (type(ri1) is np.float64 and type(ri2) is np.float64) or (type(ri1) is float and type(ri2) is float):
        return 1 if abs(abs(ri2) - abs(ri2)) <= 0.2 * ri1 else 0
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
    
    aux = random.sample(list(support_columns(df, r)), min(count, support_count(df, r)))
    return aux

# Aux(r) as defined by the paper
def auxillary(df, r, count):
    
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

# ScoreboardRH() as defined by the paper
def scoreboard_robust(df, count):
    
    aux_series = pd.Series(df.index).apply(lambda r1: auxillary(df, r1, count))
    scores = pd.DataFrame(index=df.index, columns=df.index)
    
    for r1 in tqdm(list(df.index)):
        scores.loc[r1, :] = pd.Series(df.index).apply(lambda r2: score_robust(df, aux_series.loc[r1, ~aux_series.loc[r1].isna()], r2))
            
    return scores

df3 = df.loc[:, [col for col in df.columns]]

shares = df.columns.to_series().apply(lambda x: int((list(df.loc[:, x].value_counts())[0])/len(df)*100))
roundshares = shares.apply(lambda x: True if x > 50 else False)

df4 = df.progress_apply(lambda col: col.apply(lambda x: x != col.mode()[0]) if roundshares.get(col.name, False) else col)

nn = pd.Series(df4.index).progress_apply(lambda x: max([compute_similarity(df4, x, i) for i in df4.index if i != x]))
nn.to_csv(f"{BASE_PATH}/results/nearestneighbours_presence_r4.csv")

sorted_nn = nn.reset_index().sort_values(by=0, ascending=False)
fractions = np.arange(1, len(sorted_nn) + 1) / len(sorted_nn)

plt.figure(figsize=(4, 8))
plt.plot(sorted_nn[0].values, fractions, marker='', label='Matching values')
plt.xlabel("Similarity")
plt.ylabel("Fraction of Subscribers")
plt.title("Nearest-Neighbor Similarity Distribution")
plt.legend(fontsize=10)
plt.savefig(f"{BASE_PATH}/images/nn_presence_r4.png", dpi=600, bbox_inches="tight")
