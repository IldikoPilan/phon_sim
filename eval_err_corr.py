# -*- coding: utf-8 -*-#
from __future__ import division
import sys
import codecs
import os
import numpy
activate_this = os.path.join(os.path.dirname(__file__), 'venv/bin/activate_this.py')
execfile(activate_this, dict(__file__=activate_this))
from gensim.models import FastText
from g2p_seq2seq import g2p
from g2p_seq2seq import data_utils
import phon_sim
from scipy.stats import spearmanr

def load_eval_data(data_file, print_info=False):
    """ Reads error correction data file (TAB-separated) and 
    returns a nested list of elements per row.
    """
    with codecs.open(data_file,"r", "utf-8") as f:
        rows = f.readlines()
    headers = rows[0].strip("\n").split("\t")
    error_data = []
    stats = {"SpIn-A1":0, "SpIn-A2":0, "SpIn-B1":0, "SpellEx":0}
    essays = {"A1":[], "A2":[], "B1":[]}
    for row in rows[1:]:
        cells = row.strip("\n").split("\t")
        original, corrected, source, cefr, context = tuple([cell.strip(" ") for cell in cells])
        if cefr:
            stats["SpIn-%s"%cefr] += 1
            essays[cefr].append(source)
        else:
            stats["SpellEx"] += 1
        cells[0] = original.lower()
        cells[1] = corrected.lower()
        error_data.append(cells)
    if print_info:
        print "### Statistics over L2 error data ###"
        print "Total SpIn", sum([v2 for k2, v2 in stats.items() if k2 != "SpellEx"])
        for k, v in sorted(stats.items()):
            print k, v
        print "Total", sum([v2 for k2, v2 in stats.items()])
        for k3, v3 in essays.items():
            print k3, len(set(v3)), "essays"
    return error_data

def get_related_words(token, model, topn, print_words=False):
    """
    look up the topn most similar terms to token in embedding
    and print them as a formatted list
    """
    rel_words = [(word.encode("utf-8"), sim) for word, sim in model.wv.most_similar(positive=[token], topn=topn)]    
    if print_words:
        for word, sim in rel_words:
            print '{:20} {}'.format(word.encode("utf-8"), round(sim, 3)) 
    return rel_words

def is_in_topn(eval_data, model, topn, save=False):
    result = {"SpIn":0, "SpellEx":0}
    for row in eval_data:
        original, correct, source = row[0].strip(" "), row[1].strip(" "), row[2].strip(" ")
        rel_words = get_related_words(original, model, topn)
        rel_words_w = [w.decode("utf-8") for w, sim in rel_words]
        if correct in rel_words_w:
            in_we = True
            if "SpIn" in source:
                result["SpIn"] += 1
            else:
                result["SpellEx"] += 1
        else:
            in_we = False
        if save:
            with codecs.open("topn.csv", "a", "utf-8") as f:
                rw_str = ",".join(["-".join([t[0].decode("utf-8"),str(t[1])]) for t in rel_words])
                f.write("\t".join([original, correct, str(in_we), rw_str, "\n"]))
    return result

def eval_correct_presence(eval_data, model, topns):
    """
    @ topns: number of n most similar words
    """
    for topn in topns:
        print topn, "&",
        if topn == 50:
            res_topn = is_in_topn(eval_data, model, topn, save=True) 
        else:
            res_topn = is_in_topn(eval_data, model, topn)
        spin_perc = round(res_topn["SpIn"] / 202, 3) * 100
        spellex_perc = round(res_topn["SpellEx"] / 253, 3) * 100 
        print  spin_perc, "&", spellex_perc, "&", (spin_perc+spellex_perc)/2


def compute_similarity(eval_data, model, save=False):
    for row in eval_data:
        original, correct, source = row[0], row[1], row[2]
        sim = model.wv.similarity(original, correct)
        if save:
            with codecs.open("cos_sim.csv", "a", "utf-8") as f:
                f.write("\t".join([original, correct, source, str(round(sim,3)),"\n"]))
    return sim

def compute_phon_sim(eval_data, phon_data, g2p_model_dir, save=False):
    with g2p.tf.Graph().as_default():
        g2p_model = g2p.G2PModel(g2p_model_dir)
    g2p_model.load_decode_model()
    orig_trs = g2p_model.decode([row[0] for row in eval_data]) 
    corr_trs = g2p_model.decode([row[1] for row in eval_data]) 
    phon_sims = []
    for i,row in enumerate(eval_data):
        original, correct, source = row[0], row[1], row[2]
        phon_d = phon_sim.phon_levenshtein(orig_trs[i], corr_trs[i], phon_data)
        ort_d = phon_sim.levenshtein(original, correct)
        norm_phon_sim = phon_sim.get_norm_sim(orig_trs[i], corr_trs[i], phon_d)
        norm_sim = phon_sim.get_norm_sim(original, correct, ort_d, "orth")
        if save:
            with codecs.open("phon_sim.csv", "a", "utf-8") as f:
                f.write("\t".join([original, correct, source, str(norm_sim), str(norm_phon_sim), orig_trs[i], 
                         corr_trs[i], "\n"]))
        phon_sims.append(phon_sim)
    return phon_sims

# Example calls
# data_file = "swe_L2_errors_SUBM.csv"
# eval_data = load_eval_data(data_file)

## G2P 
# model_dir = "swe_nst_g2p_2"
# r=g2p_model.decode([u"älska"]) #.decode("utf-8")
# print r

# phon_data_file = "phon_features.csv"
# phon_data = phon_sim.vectorize_phon_feats(phon_data_file)
# compute_phon_sim(eval_data, phon_data, model_dir, True) # run from terminal (encoding issues)
