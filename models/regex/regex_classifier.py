#
# This regex classifier is taken from FoldFirstAskLater project
# The code has been slighly formatted as it was originally in a pyjupiter format 
# A light class wrapper was added to fit our project strcture
#
import re

# final filter lists
hypothetical_af_search = ["whole genome shotgun sequence"] 
hypothetical_af_match = ["unnamed product", 
                         r"genome assembly, chromosome: [\w+]+", r"str. [\w+]+[\d+]+", 
                         "protein of unknwon function", 
                         "conserved domain protein, histidine-rich", r"lin[\d+]+ protein", r"orf[\w+]+( domain-containing)?( protein)?", 
                         "phage related-protein", "phage d3 protein", 
                         r"dna, contig(: [\w+]+)?", r"uncharacterized protein conserved in bacteria(, prophage-related)?", r"gene [\d+]+ protein",
                         r"gifsy-[\d+] prophage protein", r"(hypothetical )?genomic island protein",
                         r"(putative )?(conserved )?(\([a-z ]+\) )?((uncharacterized)|(hypothet(h)?ical)|(unannotated))( conserved)?( protein)?(, (pro)?phage-related)?(, isoform [AB])?",
                         r"(predicted|unannotated|uncharacterized|hypothetical|putative( |_)?)?(expressed)?(conserved)?((mobile element)|(integron protein cassette))?(( )?domain)?(constituent)?([\d+]+)?( |_)protein(_-_conserved)?", 
                         r"((hypothetical|putative|probable|conserved|uncharacterized) )?((hypothetical|putative|probable|conserved|uncharacterized) )?(bacterio|pro)?(phage)((-| )(like|related|associated))?( (hypothetical|putative|probable|conserved|uncharacterized))?( protein)?(,? [g]?p[\d+]+\b)?(, family)?(, putative)?",
                         r"hk97( family)?( phage)? protein", r"putative similar to (bacterio)?phage protein",
                         r"(putative )?uncharacterized protein(?! duf[\d+]+)( \w+)?", 
                         r"(phage( |-))?(protein )?(putative )?[g]?p[\d+]+\b(-like)?( domain(-containing)?)?(( |-)family)?( protein)?"
                        ]

# functions to check whether description is in the list of hypothetical protein annotations for AFdb annotations
def is_hypothetical_af(descr):
    hypothetical_label = False
    for filter_descr_s in hypothetical_af_search:
        if re.search(filter_descr_s, descr.strip().lower()) != None:
            hypothetical_label = True
    for filter_descr_m in hypothetical_af_match:
        if re.fullmatch(filter_descr_m, descr.strip().lower()) != None:
            hypothetical_label = True
    return hypothetical_label


# set of filters to identify low informative annotations
low_af_search = [r"duf[0-9]+", "br0599"] 
low_af_match = [r"uncharacterized (mitochondrial )?protein ((xf_1581/xf_1686)|(atmg00810-like))",
                  r"uncharacterized( membrane)? protein ([\w+]+ )?\((upf)[\d+]+( family\))", r"cson[\d+]+ protein", 
                  "putative phage-related exported protein",
                  r"((hypothetical|predicted|putative) )?(conserved )?((pro)?phage )?(cytosolic|membrane|secreted|exported|cytoplasmic)( associated)?( phage)? protein(, putative)?"
                 ]


# function to test whether an annotation is to be considered low informative
def is_low_af(descr):
    low_label = False
    for filter_descr_s in low_af_search:
        if re.search(filter_descr_s, descr.strip().lower()) != None:
            low_label = True
    for filter_descr_m in low_af_match:
        if re.fullmatch(filter_descr_m, descr.strip().lower()) != None:
            low_label = True
    return low_label


class RegexModel(object):
    def predict(self, X):
        """_summary_

        Args:
            X (iterateable): 

        Returns:
            list : predictions
        """
        preds = []
        for annotation in X:
            label = ""
            if is_hypothetical_af(annotation) == True:
                label = "uninformative"
            elif is_low_af(annotation) == True:
                label = "low"
            else:
                label = "proper"
            preds.append(label)
        return preds