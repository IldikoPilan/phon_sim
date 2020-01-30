# -*- coding: utf-8 -*-

import codecs
import os
import re
import random
import sys

import cchardet as chardet

def segment_transcription(transc_w):
    """
    Splits the phonetic transcription of a word in NST into
    individual phonemes. Maintains the original phoneme order.  
    """
    phones = ['}:', '2:', 'A:', 'e:', 'i:', 'u:', 'y:', 'o:', 'u0', '9', '@', 'd`', 't`', 
              'n`', 'l`', 's`', 'N', 'r', 'a', 'e', 'I', 'U', 'Y', 'O', 'E:', 'b', 
              'd', 'g', 'p', 't', 'k', 'f', 's', 'v', 'C', 'm', 'n', 'l', 'h', 'j', 
              'E', r'x\\', 's\''] #40 + 3 added based on what's occurring in data, C not occuring
              # Phones removed: 
              # $ syllable boundary
              # % compound boundary
              # _ phrasal verb boundary? 
              # ¤ word boundary
              # "" stress
              # duplicate }:  
              # BUG: re package will not instert a space after x\ phone, added manually 
    phon_units = []
    orig_pos = [(phone,p.start()) for phone in phones 
                       for p in re.finditer(phone,transc_w)]
    used_idx = []
    for i in sorted(range(4), reverse=True):
        for phone in phones:
            if len(phone) > i:                
                matches = re.findall(phone,transc_w)
                for match in matches: 
                    match_start = transc_w.index(match)
                    match_end = match_start+i+1
                    for phon, start in orig_pos:
                        if phon == phone and start not in used_idx:
                            phon_units.append((start,transc_w[match_start:match_end]))
                            used_idx.append(start)
                    analyzed_w = transc_w[:match_start] + transc_w[match_end:]
                    transc_w = analyzed_w
    return [pu for ix, pu in sorted(phon_units)]

def parse_nst_dict(nst_file, dict_file):
    nst_dict = []
    nst_fields = {"orthography":0, "lang_code":6, "garbage":7, "acronym":9, "transcription":11, "lemma":32} 
    # transcr. variants: 11-26, but picked only first variant
    with codecs.open(nst_file, "r", "WINDOWS-1252") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip("\n").split(";")
            if data[nst_fields["lang_code"]] == "SWE" and data[nst_fields["garbage"]] != "GARB" \
                and data[nst_fields["acronym"]] != "ACR":
                segm_transcr = " ".join(segment_transcription(data[nst_fields["transcription"]]))
                if "_" not in data[nst_fields["orthography"]]:
                    if data[nst_fields["orthography"]] == data[nst_fields["lemma"]].split("|")[0]:
                        entry = data[nst_fields["orthography"]].lower() + " " + segm_transcr
                        nst_dict.append(entry)

    print "Extracted %d items" % len(nst_dict)
    with codecs.open(dict_file, "w") as f:
        f.write("\n".join(nst_dict).encode("utf-8"))
    return nst_dict

    
def split_train_test(nst_dict_file, test_amount, remove_duplicates=True):
    " @ test_amount: percentage of data to use for testing "

    with codecs.open(nst_dict_file) as f:
        nst_dict = f.readlines()
        
    if remove_duplicates:
        nst_dict = list(set(nst_dict))

    random.seed(9)
    random.shuffle(nst_dict)
    test_size = int(len(nst_dict) / test_amount)
    print "Unique:", len(nst_dict)

    with codecs.open(nst_dict_file[:-5]+"_train.dict", "w") as f:
        f.write("".join(sorted(nst_dict[test_size:])))
        print "Train:", len(nst_dict[test_size:])

    with codecs.open(nst_dict_file[:-5]+"_test.dict", "w") as f:
        f.write("".join(sorted(nst_dict[:test_size])))
        print "Test:", len(nst_dict[:test_size])
