
from platform import python_branch
import torch
# import torch.nn.functional as F

# x = torch.rand(10, 3) # shape: seq_len, embed_dim
# l1 = torch.nn.Linear(3, 1) # shape: inp_feature_dim, n_classes
# sigmoid = torch.nn.Sigmoid()

# out = sigmoid(l1(x))
# print(torch.round(out).shape)


# input = torch.randn(16, 20) # seq_len, embed_dim
# input.unsqueeze_(0)  # 1, seq_len, embed_dim
# n_classes = 3
# n_classes = 1 if n_classes <= 2 else n_classes
# m = torch.nn.Conv2d(1, n_classes, kernel_size=(5,20), padding=(2, 0)) # in_channels=1, out_channels=n_classes, kernel_size=(window_size, embed_dim), padding=(floor(window_size/2), 0)

# out = m(input)
# print(out.shape)
# out.squeeze_(2).t_()
# if n_classes<=2:
#     print(torch.sigmoid(out)) # for binary classification, use sigmoid for inference.
# else:
#     print(torch.softmax(out, dim=1))


# n_classes=2
# input = torch.randn(16, 20) # seq_len, embed_dim
# input.unsqueeze_(0)  # 1, seq_len, embed_dim

# rnn = torch.nn.LSTM(20, n_classes, 1, batch_first=True, bidirectional=False) # embed_dim, n_classes

# out, (hn, cn) = rnn(input)
# out.squeeze_(0)
# # print(out)

# if n_classes<=2:
#     print(torch.sigmoid(out))  # for binary classification, use sigmoid for inference.
# else:
#     print(torch.softmax(out, dim=1)) # for multiclass

import numpy as np

DNAorRNA_ids = ['NUC']
peptide_ids = ['III']
metal_ids = ['LA', 'NI', '3CO', 'K', 'CR', 'ZN', 'CD', 'PD', 'TB', 'YT3', 'OS', 'EU', 'NA', 'RB', 'W', 'YB', 'HO3', 'CE', 'MN', 'TL', 'LI', 'MN3', 'AU3', 'AU', 'EU3', 'AL', '3NI', 'FE2', 'PT', 'FE', 'CA', 'AG', 'CU1', 'LU', 'HG', 'CO', 'SR', 'MG', 'PB', 'CS', 'GA', 'BA', 'SM', 'SB', 'CU', 'MO', 'CU2']
non_regular_ids = ["UUU"] + DNAorRNA_ids + peptide_ids + metal_ids # regular is small

def set_bit(x:int, seq_len, seq_pos):
    # bits: right-most pos is 0
    # string: left-most pos is 0
    return x | (1<<(seq_len-1-seq_pos))

def int_to_binary(x:int, len):
    return format(x, f"0{len}b")

def int_to_binaryarray(x:int, len):
    x = format(x, f"0{len}b")
    return np.array([int(c) for c in x])

def int_to_hex(x:int, len):
    return format(x, f"X")

def print_seq_and_label(seq, int_label):
    seq_len = len(seq)
    print(f"seq: {seq}")
    print(f"bin: {int_to_binary(int_label, seq_len)}")
    # print(f"arr: {int_to_binaryarray(int_label, seq_len)}")
    # print(f"hex: {int_to_hex(int_label, seq_len)}")
    # print(f"int: {int_label}")

def generate_label(seq:str, binding_sites:str, do_print=False):
    seq_len = len(seq)
    int_label = int.from_bytes(bytearray(seq_len), "big")

    binding_sites = binding_sites.rstrip().split() # list of binding sites, 1-indexed according to BioLip
    for binding_site in binding_sites:
        aa, pos = binding_site[0], int(binding_site[1:])-1 # 0-indexed
        int_label = set_bit(int_label, seq_len, pos)
    
    if do_print: print_seq_and_label(seq, int_label)
    return int_label

# sample usage
# binding_sites = "H3 Y9 Y13 Y14 E23 I26 G52 S53 T56 Q59 Y75 N79 L153 D156 A157 A158" # 1-indexed according to BioLip
# seq = "LSHFNPRDYLEKYYKFGSRHSAESQILKHLLKNLFKIFCLDGVKGDLLIDIGSGPTIYQLLSACESFKEIVVTDYSDQNLQELEKWLKKEPAAFDWSPVVTYVCDLEGNRVKGPEKEEKLRQAVKQVLKCDVTQSQPLGAVPLPPADCVLSTLCLDAACPDLPTYCRALRNLGSLLKPGGFLVIMDALKSSYYMIGEQKFSGREAVEAAVKEAGYTIEWFEVIGLFSLVARKL"
# generate_label(seq, binding_sites, True)
import os.path as osp
class ProteinBindingSite(object):
    def __init__(self, pdb_chain_id:str, uniprot_id:str, seq:str, DNAorRNA_binding_sites:str, peptide_binding_sites:str, metal_binding_sites:str, regular_binding_sites:str):
        super(ProteinBindingSite, self).__init__()
        self.pdb_chain_id = pdb_chain_id
        self.uniprot_id = uniprot_id
        self.seq = seq
        self.DNAorRNA_binding_sites = [] if len(DNAorRNA_binding_sites)==0 else [DNAorRNA_binding_sites]
        self.peptide_binding_sites = [] if len(peptide_binding_sites)==0 else [peptide_binding_sites]
        self.metal_binding_sites = [] if len(metal_binding_sites)==0 else [metal_binding_sites]
        self.regular_binding_sites = [] if len(regular_binding_sites)==0 else [regular_binding_sites]
    
    def update(self, DNAorRNA_binding_sites:str, peptide_binding_sites:str, metal_binding_sites:str, regular_binding_sites:str):
        if len(DNAorRNA_binding_sites)>0: self.DNAorRNA_binding_sites.append(DNAorRNA_binding_sites)
        if len(peptide_binding_sites)>0: self.peptide_binding_sites.append(peptide_binding_sites)
        if len(metal_binding_sites)>0: self.metal_binding_sites.append(metal_binding_sites)
        if len(regular_binding_sites)>0: self.regular_binding_sites.append(regular_binding_sites)
        return self

    def save(self, filepath):
        f = open(filepath, "a")
        tab_sep_DNAorRNA_binding_sites = "\t".join(self.DNAorRNA_binding_sites)
        tab_sep_peptide_binding_sites = "\t".join(self.peptide_binding_sites)
        tab_sep_metal_binding_sites = "\t".join(self.metal_binding_sites)
        tab_sep_regular_binding_sites = "\t".join(self.regular_binding_sites)
        f.write(f">{self.pdb_chain_id} {self.uniprot_id}|{tab_sep_DNAorRNA_binding_sites}|{tab_sep_peptide_binding_sites}|{tab_sep_metal_binding_sites}|{tab_sep_regular_binding_sites}\n")
        f.write(self.seq)
        f.write("\n")


    def __str__(self) -> str:
        return f"pdb_chain_id: {self.pdb_chain_id}\nuniprot_id: {self.uniprot_id}\nseq: {self.seq}\nDNAorRNA_binding_sites: {self.DNAorRNA_binding_sites}\n" +\
               f"peptide_binding_sites: {self.peptide_binding_sites}\nmetal_binding_sites: {self.metal_binding_sites}\nregular_binding_sites: {self.regular_binding_sites}\n" 



inp_filepath = "data/downloads/biolip_2022-03-30_nr/BioLiP_2022-03-30_nr.txt"
f = open(inp_filepath, "r")

data = {}
for i, line in enumerate(f.readlines()):
    line = line.rstrip()
    line_items = [''] + line.split("\t") # to keep the column no similar to the "https://zhanggroup.org/BioLiP/download/readme.txt"
    # print(line_items)
    pdb_id = line_items[1]
    pdb_chain = line_items[2]
    ligand_id = line_items[5]
    ligand_chain_= line_items[6]
    ligand_serian_num = line_items[7]
    binding_sites = line_items[9] # 1-indexed according to BioLip
    uniprot_id = line_items[18]
    seq =  line_items[20]
    # int_lable = generate_label(seq, binding_sites, do_print=False)

    DNAorRNA_binding_sites, peptide_binding_sites, metal_binding_sites, regular_binding_sites = "", "", "", ""

    if ligand_id in DNAorRNA_ids:
        DNAorRNA_binding_sites = binding_sites
    elif ligand_id in peptide_ids:
        peptide_binding_sites = binding_sites
    elif ligand_id in metal_ids:
        metal_binding_sites = binding_sites
    elif ligand_id not in non_regular_ids:        
        regular_binding_sites = binding_sites
    else:
        raise Exception(f"Ligand_id={ligand_id} not expected.")

    pdb_chain_id = pdb_id + pdb_chain

    if pdb_chain_id not in data: # new protein point
        data[pdb_chain_id] = ProteinBindingSite(pdb_chain_id, uniprot_id, seq, DNAorRNA_binding_sites, peptide_binding_sites, metal_binding_sites, regular_binding_sites)
    else:
        proteinBindingSite = data[pdb_chain_id]
        proteinBindingSite.update(DNAorRNA_binding_sites, peptide_binding_sites, metal_binding_sites, regular_binding_sites)

    # print(data[pdb_chain_id])
    
    # data[pdb_chain_id]["all_binding_sites"] = existing_datam["DNAorRNA_binding_sites"] | existing_datam["peptide_binding_sites"] | existing_datam["metal_binding_sites"] | existing_datam["regular_binding_sites"]
    if i==10: break
print(len(data))

for pdb_chain_id, proteinBindingSite in data.items():
    proteinBindingSite.save("data/y.fasta")