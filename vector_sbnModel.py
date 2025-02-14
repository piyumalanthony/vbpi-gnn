import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bitarray import bitarray
from ete3 import Tree
from utils import BitArray, logsumexp
import pdb



class ParamParser(object):
    def __init__(self):
        self.start_and_end = {}
        self.num_params = 0
        self.num_params_in_dicts = 0
        self.dict_name_list = []
        # self.dict_len = []
        
    def add_item(self, name):
        start = self.num_params
        self.num_params += 1
        self.start_and_end[name] = start
        # self.start_and_end[name] = (start, self.num_params)
        
    def check_item(self, name):
        return name in self.start_and_end
        
    def add_dict(self, name, record_name=True):
        start = self.num_params_in_dicts
        self.num_params_in_dicts = self.num_params
        self.start_and_end[name] = (start, self.num_params)
        if record_name:
            self.dict_name_list.append(name)
        # self.dict_len.append(self.num_params - start)
    
    def get(self, tensor, name):
        start, end = self.start_and_end[name]
        return tensor[start:end]
        
    def get_index_or_slice(self, name):
        index_or_slice = self.start_and_end[name]
        if isinstance(index_or_slice, tuple):
            start, end = index_or_slice
            return list(range(start, end))
        else:
            return index_or_slice
                
    def get_index(self, name):
        return self.start_and_end[name]
        

class SBN(nn.Module):
    """
    Vectorized Subsplit Bayesian Networks (SBNs) Module.
    
    """
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict):
        super().__init__()
        self.taxa, self.ntaxa = taxa, len(taxa)
        self.toBitArr = BitArray(taxa)
        self.rootsplit_supp_dict = rootsplit_supp_dict
        self.subsplit_supp_dict = subsplit_supp_dict

        print("Rootsplit support dict:", self.rootsplit_supp_dict)
        print("Subsplit support dict:", self.subsplit_supp_dict)

        # Note: rootsplit_supp_dict only has rootsplicts with the keys with size of n_tips
        # Rootsplit support dict: OrderedDict([('011111', 3.0), ('010000', 3.0), ('001111', 2.0), ('001110', 1.0), ('000001', 3.0),...
        # Note: subsplit_supp_dict has two kind of keys. 1. parent key (size of 2*n_tips) 2. child key (size of n_tips)
        # Subsplit support dict: OrderedDict([('100000011111', OrderedDict([('001111', 2.0), ('001000', 1.0)])), ('010000101111', OrderedDict([('001111', 2.0), ('000110', 1.0)])), ....

        self.CPDParser = ParamParser()
        for split in self.rootsplit_supp_dict:
            self.CPDParser.add_item(split)
        self.CPDParser.add_dict('rootsplit', record_name=False)
        self.rs_len = len(self.rootsplit_supp_dict)

        # print(self.CPDParser.start_and_end)
                
        ss_mask, ss_max_len = [], 0
        for parent in self.subsplit_supp_dict:
            ss_len = len(self.subsplit_supp_dict[parent])
            if ss_len > 1:
                for child in self.subsplit_supp_dict[parent]:
                    self.CPDParser.add_item(parent + child)
                self.CPDParser.add_dict(parent)
                ss_mask.append(torch.ones(ss_len, dtype=torch.uint8))
                ss_max_len = max(ss_max_len, ss_len)

        print("\n")
        print("CPD parser:", self.CPDParser.start_and_end)
                        
        self.ss_mask = torch.stack([F.pad(mask, (0, ss_max_len - mask.size(0)), 'constant', 0) for mask in ss_mask], dim=0)
        # self.ss_mask = self.ss_mask.bool()
        
                
        self.CPD_params = nn.Parameter(torch.zeros(self.CPDParser.num_params), requires_grad=True)

        # print("CPDParser num_params:", self.CPDParser.num_params)
        self.idx_map = np.append(np.arange(self.CPDParser.num_params), [-2,-1])
        # print("idx_map:", self.idx_map)
        
        # self.rs = nn.Parameter(torch.zeros(len(self.rootsplit_supp_dict)), requires_grad=True)
        # self.rs_CPDs = F.softmax(self.rs, 0)
        self.rs_CPDs = F.softmax(self.CPDParser.get(self.CPD_params, 'rootsplit'), 0)
        self.rs_map = {split: i for i, split in enumerate(self.rootsplit_supp_dict.keys())}
        self.rs_reverse_map = {i: split for i, split in enumerate(self.rootsplit_supp_dict.keys())}
        
        
        self.subsplit_parameter_set = set(self.CPDParser.dict_name_list)
        self.ss_name_map = {parent: i for i, parent in enumerate(self.CPDParser.dict_name_list)}
        # self.subsplit_parameter_dict = nn.ParameterDict({parent: nn.Parameter(torch.zeros(len(self.subsplit_supp_dict[parent])), requires_grad=True) for parent in self.subsplit_supp_dict.keys() if len(self.subsplit_supp_dict[parent]) > 1})
                
        self.ss_map = {}
        self.ss_reverse_map = {}
        # self.ss_CPDs = {parent: torch.tensor([1.0]).detach() for parent in self.subsplit_supp_dict if len(self.subsplit_supp_dict[parent]) == 1}
        for parent in self.subsplit_supp_dict:
            self.ss_map[parent] = {child: i for i, child in enumerate(self.subsplit_supp_dict[parent].keys())}
            self.ss_reverse_map[parent] = {i: child for i, child in enumerate(self.subsplit_supp_dict[parent].keys())} 
            # if parent in self.subsplit_parameter_dict:
            #     self.ss_CPDs[parent] = F.softmax(self.subsplit_parameter_dict[parent], 0)
        
        # self.CPDs = torch.cat((self.rs_CPDs, self.update_subsplit_CPDs()))
        ss_CPDs, self.ss_masked_CPDs = self.update_subsplit_CPDs()

        # print("ss_CPDs:", ss_CPDs)
        # print("ss_masked_CPDs:", self.ss_masked_CPDs)

        self.CPDs = torch.cat((self.rs_CPDs, ss_CPDs))
        self.one_tensor = torch.tensor([1.0])
            
    def update_rootsplit_CPDs(self):
        # self.rs_CPDs = F.softmax(self.rs, 0)
        self.rs_CPDs = F.softmax(self.CPDParser.get(self.CPD_params, 'rootsplit'), 0)
        if torch.isnan(self.rs_CPDs).any():
            raise Exception('Invalid rootsplit probability! Check self.rs:(max {:.4f}, min {:.4f})'.format(np.max(self.rs.detach().numpy()), np.min(self.rs.detach().numpy())))
        
    def update_subsplit_CPDs(self):
        # ToDO: Here the code is updated to remove depreicated code which does not support masks except boo.
        temp_mat = torch.zeros(self.ss_mask.size())
        temp_mat.masked_scatter_(self.ss_mask.to(torch.bool), self.CPD_params[self.rs_len:])
        masked_temp_mat = temp_mat.masked_fill((1-self.ss_mask).to(torch.bool), -float('inf'))
        masked_CPDs = F.softmax(masked_temp_mat, dim=1)

        print("Inside update subsplits function")
        print("CPDs:",  self.CPD_params[self.rs_len:])
        print("rslen:",self.rs_len)
        print("ss_len:",self.CPD_params.size())
        print("ss bool mask:", self.ss_mask.to(torch.bool))
        print("CPDs:", self.CPD_params[self.rs_len:])
        # print("CPDs:", masked_CPDs)
        # print("mask ss:", self.ss_mask)
        #
        # print("mask out ss:", masked_CPDs.masked_select(self.ss_mask.to(torch.bool)))
        
        return masked_CPDs.masked_select(self.ss_mask.to(torch.bool)), masked_CPDs
        
    
    def update_CPDs(self):
        self.update_rootsplit_CPDs()
        ss_CPDs, self.ss_masked_CPDs = self.update_subsplit_CPDs()
        self.CPDs = torch.cat((self.rs_CPDs, ss_CPDs))
        
    
    def check_parent_child(self, parent, child=None):
        if parent not in self.ss_map:
            return False
        else:
            if child and child not in self.ss_map[parent]:
                return False
        return True
        
    def node_subsplit_idxes_update(self, node_subsplit_idxes, ss_parent, ss_child):
        if not self.check_parent_child(ss_parent, ss_child):
            node_subsplit_idxes.append(-1)
        else:
            ss_name = ss_parent + ss_child
            if self.CPDParser.check_item(ss_name):
                node_subsplit_idxes.append(self.CPDParser.get_index(ss_name))
            else:
                # Note: This is because as for a optimization, we don't include ss_len = 1 items in line 84 in init().
                # Therefore, all probabilities for theses sub splits should be 1.
                node_subsplit_idxes.append(-2)
                     
        
    def get_rootsplit_CPDs(self, rootsplit):
        return self.rs_CPDs[self.rs_map[rootsplit]]
        
    def get_subsplit_CPDs(self, parent, child=None):
        if child:
            return self.CPDParser.get(self.CPDs, parent)[self.ss_map[parent][child]]
        else:
            if parent in self.subsplit_parameter_set:
                return self.CPDParser.get(self.CPDs, parent)
            else:
                # return self.ss_CPDs[parent]
                return self.one_tensor
            
    
    def sample_tree(self, rooted=False):
        root = Tree()
        node_split_stack = [(root, '0'*self.ntaxa + '1'*self.ntaxa)]
        for i in range(self.ntaxa-1):
            node, split_bitarr = node_split_stack.pop()
            parent_clade_bitarr = bitarray(split_bitarr[self.ntaxa:])
            node.clade_bitarr = parent_clade_bitarr
            node.split_bitarr = min([parent_clade_bitarr, ~parent_clade_bitarr]).to01()
            if node.is_root():
                split_prob = self.rs_CPDs
                # split = self.rs_reverse_map[np.random.choice(len(split_prob), p=split_prob)]
                split = self.rs_reverse_map[torch.multinomial(split_prob, 1).item()]
            else:
                split_prob = self.get_subsplit_CPDs(split_bitarr)
                # split = self.ss_reverse_map[split_bitarr][np.random.choice(len(split_prob), p=split_prob)]
                split = self.ss_reverse_map[split_bitarr][torch.multinomial(split_prob, 1).item()]

            # Note: give the other clade or bipartition
            comp_split = (parent_clade_bitarr ^ bitarray(split)).to01()
            
            c1 = node.add_child()
            c2 = node.add_child()
            ## Note: Add for next iteration to further get subsplits
            if split.count('1') > 1:
                node_split_stack.append((c1, comp_split + split))
            else:
                c1.name = self.taxa[split.find('1')]
                c1.clade_bitarr = bitarray(split)
                c1.split_bitarr = min([c1.clade_bitarr, ~c1.clade_bitarr]).to01()
            ## Note: This is other subsplit to consider given the parent. This works because as we take parent_clade_bitarr = bitarray(split_bitarr[self.ntaxa:])
            ## which is always second half of the bitarray
            if comp_split.count('1') > 1:
                node_split_stack.append((c2, split + comp_split))
            else:
                c2.name = self.taxa[comp_split.find('1')]
                c2.clade_bitarr = bitarray(comp_split)
                c2.split_bitarr = min([c2.clade_bitarr, ~c2.clade_bitarr]).to01()
        
        if not rooted:
            root.unroot()
        
        return root        
        

    def grab_subsplit_idxes(self, tree):
        """
        Traverse the tree topology to grab the indices for parent-child subsplit pairs (PCSPs).
        This is a two-pass algorithm that enjoys a linear time complexity.
        
        """

        """
        Handling Bipartition Bit Arrays
        
        The bipartition bit arrays (clade_bitarr):
        
        Encode the evolutionary relationships between taxa.
        Need to be combined and propagated differently in each direction:
        Bottom-Up: Bipartitions are computed from children and propagated to the parent.
        Top-Down: Bipartitions are derived from the parent and sibling relationships and passed to the children.
        These computations are direction-dependent, necessitating two distinct passes.
        
        """
        # Note: This is needed as the we need specific subsplits that are available in a given tree.

        # print("Inside grab_subsplit_idxes")
        for node in tree.traverse("postorder"):
            if not node.is_root():
                node.leaf_to_root_subsplit_idxes = []
                if not node.is_leaf():
                    node.leaf_to_root_child_subsplit_idxes = []
                    for child in node.children:
                        node.leaf_to_root_child_subsplit_idxes.extend(child.leaf_to_root_subsplit_idxes)
                    node.leaf_to_root_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                    node.leaf_to_root_subsplit_idxes.extend(node.leaf_to_root_child_subsplit_idxes)
                    
                    if not node.up.is_root():
                        comb_parent_bipart_bitarr = node.get_sisters()[0].clade_bitarr + node.clade_bitarr
                        ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.leaf_to_root_bipart_bitarr.to01()
                        self.node_subsplit_idxes_update(node.leaf_to_root_subsplit_idxes, ss_parent, ss_child)

                # print(node.leaf_to_root_subsplit_idxes)
                        
        
        subsplit_idxes_list = []          
        for node in tree.traverse("preorder"):
            if not node.is_root():
                node.root_to_leaf_subsplit_idxes = []
                if node.up.is_root():
                    node.root_to_leaf_bipart_bitarr = min(sister.clade_bitarr for sister in node.get_sisters())
                    for sister in node.get_sisters():
                        if not sister.is_leaf():
                            node.root_to_leaf_subsplit_idxes.extend(sister.leaf_to_root_subsplit_idxes)
                            comb_parent_bipart_bitarr = ((~node.clade_bitarr) ^ sister.clade_bitarr) + sister.clade_bitarr
                            ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), sister.leaf_to_root_bipart_bitarr.to01()
                            self.node_subsplit_idxes_update(node.root_to_leaf_subsplit_idxes, ss_parent, ss_child)
                else:
                    sister = node.get_sisters()[0]
                    node.root_to_leaf_bipart_bitarr = min(sister.clade_bitarr, ~node.up.clade_bitarr)
                    node.root_to_leaf_subsplit_idxes.extend(node.up.root_to_leaf_subsplit_idxes)
                    comb_parent_bipart_bitarr = sister.clade_bitarr + ~node.up.clade_bitarr
                    ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.up.root_to_leaf_bipart_bitarr.to01()
                    self.node_subsplit_idxes_update(node.root_to_leaf_subsplit_idxes, ss_parent, ss_child)
                    
                    if not sister.is_leaf():
                        node.root_to_leaf_subsplit_idxes.extend(sister.leaf_to_root_child_subsplit_idxes)
                        comb_parent_bipart_bitarr = ~node.up.clade_bitarr + sister.clade_bitarr
                        ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), sister.leaf_to_root_bipart_bitarr.to01()
                        self.node_subsplit_idxes_update(node.root_to_leaf_subsplit_idxes, ss_parent, ss_child)
                
                parent_bipart_bitarr = min(node.clade_bitarr, ~node.clade_bitarr)
                if parent_bipart_bitarr.to01() not in self.rs_map:
                    node_subsplit_idxes = [-1]
                else:
                    node_subsplit_idxes = [self.CPDParser.get_index(parent_bipart_bitarr.to01())]
                    
                if not node.is_leaf():
                    node_subsplit_idxes.extend(node.leaf_to_root_child_subsplit_idxes)
                    comb_parent_bipart_bitarr = ~node.clade_bitarr + node.clade_bitarr
                    ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.leaf_to_root_bipart_bitarr.to01()
                    self.node_subsplit_idxes_update(node_subsplit_idxes, ss_parent, ss_child)
                
                node_subsplit_idxes.extend(node.root_to_leaf_subsplit_idxes)
                comb_parent_bipart_bitarr = node.clade_bitarr + ~node.clade_bitarr
                ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.root_to_leaf_bipart_bitarr.to01()
                self.node_subsplit_idxes_update(node_subsplit_idxes, ss_parent, ss_child)
                
                subsplit_idxes_list.append(node_subsplit_idxes)                              
        
        return subsplit_idxes_list
        

    def loglikelihood(self, tree, no_clade_bitarr=True):
        copy_tree = tree.copy()
        if no_clade_bitarr:
            for node in copy_tree.traverse('postorder'):
                if not node.is_root():
                    node.clade_bitarr = self.toBitArr.from_clade(node.get_leaf_names())  
                        
        with torch.no_grad():
            logprob = self.forward(copy_tree)
        return logprob.item()
        
        
    def forward(self, tree, return_idxes_list=False):
        subsplit_idxes_list = self.grab_subsplit_idxes(tree)

        # Note: For CPDs torch.tensor([1.0, 0.0]) added to represent subsplits possibility exits in the parametrization given parent
        # and subsplits not possible as the parent also does not exits. Refer to the node_subsplit_idxes_update() function.
        CPDs = torch.cat((self.CPDs, torch.tensor([1.0, 0.0])))
        mapped_idxes_list = torch.LongTensor(subsplit_idxes_list)

        # Note; logsumexp first exponenciate and then take the sum. This is equal to taking the sum of CPDs
        # This is a rooted tree where all the rootsplit summation should be calculated and sum-up for ll
        # Note: If you need clarification run sbn_param_check.py for process.
        if not return_idxes_list:
            return CPDs[mapped_idxes_list].clamp(1e-06).log().sum(1).logsumexp(0)
        else:
            return CPDs[mapped_idxes_list].clamp(1e-06).log().sum(1).logsumexp(0), subsplit_idxes_list