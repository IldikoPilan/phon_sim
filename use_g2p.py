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
            with codecs.open("results/phon_sim.csv", "a", "utf-8") as f:
                f.write("\t".join([original, correct, source, str(norm_sim), str(norm_phon_sim), orig_trs[i], 
                         corr_trs[i], "\n"]))
        phon_sims.append(phon_sim)
    return phon_sims