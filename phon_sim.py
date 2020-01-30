# -*- coding: utf-8 -*-

from __future__ import division
import codecs
from scipy.spatial import distance
from difflib import SequenceMatcher

def vectorize_phon_feats(data_file):
    """ Loads phonetic data from a csv file containing phonemes encoded in IPA, SAMPA
    and their phonetic features based on Hayes. (2009). Introductory Phonology.
    Returns a dictonary object wiht SAMPA representations as keys and a list of 
    mapped feature values. 
    """
    with codecs.open(data_file, "r", "utf-8") as f:
        rows = f.readlines()
    value_mapping = {"+":1, "0":0, "-":-1}
    phon_data = {}
    headers = rows[0].strip("\n").split(",")
    for row in rows[1:]:
        cells = row.strip("\n").split(",")
        mapped_vals = [value_mapping[feature_val] for feature_val in cells[2:]]
        phon_data[cells[1]] = mapped_vals
    return phon_data

def get_phon_dist(phon1, phon2, phon_data):
    """ Computes the distance between two phonemes based on the proportion
    of features that differ, excluding features that are 0 (irrelevant) for 
    both sounds compared.
    @ phon1, phon2: SAMPA representation of a phoneme 
    @ phon_data:    a dictionary of SAMPA phonemes as values and a list of 
                    phonological feature values.
    """
    if phon1 and phon2:
        nr_disagr = 0
        phon1 = phon1.strip(":")
        phon2 = phon2.strip(":")
        nr_relevant_feats = len([feat for feat in zip(phon_data[phon1],phon_data[phon2]) if feat != (0,0)])
        for i, feat_val in enumerate(phon_data[phon1]):
            if feat_val != phon_data[phon2][i]:
                nr_disagr += 1
        dist =  round(nr_disagr / nr_relevant_feats, 2) 
    else:
        dist = 1
    return dist

def levenshtein(s, t):
        """ Wikipedia version (iterative with two matrix rows). """
        # Source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(t)]

def phon_levenshtein(s, t, phon_data):
        """ Modified based on Wikipedia version (Iterative with two matrix rows). 
        Arguments 's' and 't' are source and target SAMPA transcriptions.
        """
        s = s.split(" ")
        t = t.split(" ")
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                if s[i] == t[j]:
                    cost = 0
                else:
                    # compute phon distance
                    cost = 0 if s[i] == t[j] else get_phon_dist(s[i],t[j],phon_data)
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(t)]

def get_norm_sim(s, t, dist, type="phon"):
    """ Normalize distance by the maximum possible distance (= length
        of the longer word) and convert it into a similarity score.
    """

    if type=="phon":
        max_dist = max([len(s.split(" ")), len(t.split(" "))])
    else:
        max_dist = max([len(s), len(t)])
    return round((1 - (dist / max_dist)), 3)


# Test runs

# phon_data_file = "phon_features.csv"
# phon_data = vectorize_phon_feats(phon_data_file)
# get_phon_dist("t", "a", phon_data)

# s = "u0 N d U m k U t`"
# t = "u0 N d U m s k U t`"
# s_ort = "ungdomkort"
# t_ort = "ungdomskort"

# lev_d = levenshtein(s, t)
# phon_lev_d = phon_levenshtein(s, t, phon_data)
# print "\tLevD:", lev_d, "\t\t", get_norm_sim(s_ort, t_ort, lev_d, "ort")
# print "\tLevD-ph:", phon_lev_d, "\t", get_norm_sim(s, t, phon_lev_d)
