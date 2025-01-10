import numpy as np
import copy
from Bio import Phylo
from io import StringIO
# from cStringIO import StringIO
from ete3 import Tree
from bitarray import bitarray
from treeManipulation import init, namenum, nametaxon
from collections import defaultdict, OrderedDict
from vector_sbnModel import *

class BitArray(object):
    def __init__(self, taxa):
        self.taxa = taxa
        self.ntaxa = len(taxa)
        self.map = {taxon: i for i, taxon in enumerate(taxa)}

    def combine(self, arrA, arrB):
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA

    def merge(self, key):
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])

    def decomp_minor(self, key):
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))

    def minor(self, arrA):
        return min(arrA, ~arrA)

    def from_clade(self, clade):
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))

def get_support_from_mcmc(taxa, tree_dict_total, tree_names_total, tree_wts_total=None):
    rootsplit_supp_dict = OrderedDict()
    subsplit_supp_dict = OrderedDict()
    toBitArr = BitArray(taxa)
    for i, tree_name in enumerate(tree_names_total):
        tree = tree_dict_total[tree_name]
        wts = tree_wts_total[i] if tree_wts_total else 1.0
        nodetobitMap = {node:toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        for node in tree.traverse('levelorder'):
            if not node.is_root():
                rootsplit = toBitArr.minor(nodetobitMap[node]).to01()
                # rootsplit_supp_dict[rootsplit] += wts
                if rootsplit not in rootsplit_supp_dict:
                    rootsplit_supp_dict[rootsplit] = 0.0
                rootsplit_supp_dict[rootsplit] += wts
                if not node.is_leaf():
                    # Note: For internal branches
                    child_subsplit = min([nodetobitMap[child] for child in node.children]).to01()
                    for sister in node.get_sisters():
                        # Note: For Nodes with multiple sisters
                        parent_subsplit = (nodetobitMap[sister] + nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                    if not node.up.is_root():
                        # Note: This is needed to added as each node at least corresponds to two sub splits as it connects
                        # three branches
                        parent_subsplit = (~nodetobitMap[node.up] + nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

                    # Note: This is needed to added as each node at least corresponds to two sub splits as it connects
                    # three branches (Remaining part of the above)
                    parent_subsplit = (~nodetobitMap[node] + nodetobitMap[node]).to01() # direction 1 for this case
                    if parent_subsplit not in subsplit_supp_dict:
                        subsplit_supp_dict[parent_subsplit] = OrderedDict()
                    if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                        subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                    subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

                if not node.up.is_root():
                    # Note: Generally a internal node or leaf node has 1 sister. But for cases like polytomies and
                    # root of a unrooted tree has children which have more than 1 sister.
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                child_subsplit = bipart_bitarr.to01()
                if not node.is_leaf():
                    for child in node.children:
                        parent_subsplit = (nodetobitMap[child] + ~nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

                parent_subsplit = (nodetobitMap[node] + ~nodetobitMap[node]).to01() # direction 2 for this case
                if parent_subsplit not in subsplit_supp_dict:
                    subsplit_supp_dict[parent_subsplit] = OrderedDict()
                if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                    subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

    return rootsplit_supp_dict, subsplit_supp_dict

def summary_raw(file_path, truncate=None, hpd=0.95, n_rep=10):
    tree_dict_total = {}
    tree_id_set_total = set()
    tree_names_total = []
    n_samp_tree = 0


    tree_dict_rep, tree_names_rep, tree_wts_rep = get_tree_list_raw(file_path, truncate=truncate, hpd=hpd)
    for j, name in enumerate(tree_names_rep):
        tree_id = tree_dict_rep[name].get_topology_id()
        if tree_id not in tree_id_set_total:
            n_samp_tree += 1
            tree_names_total.append('tree_{}'.format(n_samp_tree))
            tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]
            tree_id_set_total.add(tree_id)

    return tree_dict_total, tree_names_total

def get_tree_list_raw(filename, burnin=0, truncate=None, hpd=0.95):
    tree_dict = {}
    tree_wts_dict = defaultdict(float)
    tree_names = []
    i, num_trees = 0, 0
    with open(filename, 'r') as input_file:
        while True:
            line = input_file.readline()
            if line == "":
                break
            num_trees += 1
            if num_trees < burnin:
                continue
            tree = Tree(line.strip())
            tree_id = tree.get_topology_id()
            if tree_id not in tree_wts_dict:
                tree_name = 'tree_{}'.format(i)
                tree_dict[tree_name] = tree
                tree_names.append(tree_name)
                i += 1
            tree_wts_dict[tree_id] += 1.0

            if truncate and num_trees == truncate + burnin:
                break
    tree_wts = [tree_wts_dict[tree_dict[tree_name].get_topology_id()]/(num_trees-burnin) for tree_name in tree_names]
    if hpd < 1.0:
        ordered_wts_idx = np.argsort(tree_wts)[::-1]
        cum_wts_arr = np.cumsum([tree_wts[k] for k in ordered_wts_idx])
        cut_at = next(x[0] for x in enumerate(cum_wts_arr) if x[1] > hpd)
        tree_wts = [tree_wts[k] for k in ordered_wts_idx[:cut_at]]
        tree_names = [tree_names[k] for k in ordered_wts_idx[:cut_at]]

    return tree_dict, tree_names, tree_wts

if __name__ == '__main__':
    file_path_for_trees = "/home/piyumal/PHD/Implementation/MLPhylogenetics/vbpi-gnn/data/SBN_param_test_data/sample.data"
    tree_dict_total, tree_names_total = summary_raw(file_path_for_trees)
    print(tree_names_total)

    taxa = ['A', 'B', 'C', 'D', 'E', 'F']

    rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_total, tree_names_total)

    print(rootsplit_supp_dict)
    print("\n")
    print(subsplit_supp_dict)
    print(len(rootsplit_supp_dict))
    print(len(subsplit_supp_dict))

    print("\n")

    sbn = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict)
    # print(sbn.CPDParser)

    sample_tree = sbn.sample_tree()
    print(sample_tree)
    for node in sample_tree.traverse("preorder"):
        print(node)
        print(node.clade_bitarr)
        print(node.split_bitarr)

    ll = sbn.loglikelihood(sample_tree)
    print("log-likelihood of sample tree:", ll)

