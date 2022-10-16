import sys
sys.path.append("../binding_sites")
import os.path as osp
import numpy as np
import random
# random.seed(1)
import math
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from classes.Embedding import SequenceEmbeddingLayer
import utils as Utils

def get_binding_pos_and_regions(seq, binding_label, window_size=9):
    # binding pos will be in the middle in the window
    seq_len = len(seq)
    binding_pos_set = set()
    bindng_region_set = set()
    for pos, aa in enumerate(binding_label):
        if aa!=".":
            binding_pos_set.add(pos)
            w = math.floor(window_size/2)
            for reg in range(pos-w, pos+w+1):
                if reg>=0 and reg<seq_len: # pos will not be outside of seq-len
                    bindng_region_set.add(reg)
    
    return binding_pos_set, bindng_region_set # unsorted


def sample_residue_binding_pos(seq, binding_label, n_samples, sample_type="binding_pos"):
    # sample_type: binding_pos, binding_region, nonbinding_pos, nonbinding_region
    if len(seq)!=len(binding_label):
        raise Exception("len(seq)!=len(binding_label)")

    binding_pos_set, bindng_region_set = get_binding_pos_and_regions(seq, binding_label, window_size=9)
    # print(binding_pos_set, bindng_region_set, sep="\n")

    pos_set = set([i for i in range(0, len(seq))])
    # nonbinding_pos_set = pos_set-binding_pos_set # not in use currently
    nonbinding_region_set = pos_set-bindng_region_set
    # print(non_binding_pos_set, non_binding_region_set, sep="\n")

    samples = []
    if sample_type=="binding_pos": 
        if n_samples == "all" or n_samples >= len(binding_pos_set): samples = list(binding_pos_set)
        else: samples = random.sample(list(binding_pos_set), n_samples)
    elif sample_type=="nonbinding_region": 
        if n_samples >= len(nonbinding_region_set): samples = list(nonbinding_region_set)
        else: samples = random.sample(list(nonbinding_region_set), n_samples)
    # not in use currently
    # elif sample_type=="binding_region": 
    #     if n_samples >= len(bindng_region_set): samples = list(bindng_region_set)
    #     else: samples = random.sample(list(bindng_region_set), n_samples)
    # elif sample_type=="nonbinding_pos": 
    #     if n_samples >= len(nonbinding_pos_set): samples = list(nonbinding_pos_set)
    #     else: samples = random.sample(list(nonbinding_pos_set), n_samples)
    
    # print(samples)
    return samples # list
    
    
# seq="MGEVVRLTNSSTGGPVFVYVKDGKIIRMTPMDFDDAVDAPSWKIEARGKTFTPPRKTSIAPYTAGFKSMIYSDLRIPYPMKRKSFDPNGERNPQLRGAGLSKQDPWSDYERISWDEATDIVVAEINRIKHAYGPSAILSTPSSHHMWGNVGYRHSTYFRFMNMMGFTYADHNPDSWEGWHWGGMHMWGFSWRLGNPEQYDLLEDGLKHAEMIVFWSSDPETNSGIYAGFESNIRRQWLKDLGVDFVFIDPHMNHTARLVADKWFSPKIGTDHALSFAIAYTWLKEDSYDKEYVAANAHGFEEWADYVLGKTDGTPKTCEWAEEESGVPACEIRALARQWAKKNTYLAAGGLGGWGGACRASHGIEWARGMIALATMQGMGKPGSNMWSTTQGVPLDYEFYFPGYAEGGISGDCENSAAGFKFAWRMFDGKTTFPSPSNLNTSAGQHIPRLKIPECIMGGKFQWSGKGFAGGDISHQLHQYEYPAPGYSKIKMFWKYGGPHLGTMTATNRYAKMYTHDSLEFVVSQSIWFEGEVPFADIILPACTNFERWDISEFANCSGYIPDNYQLCNHRVISLQAKCIEPVGESMSDYEIYRLFAKKLNIEEMFSEGKDELAWCEQYFNATDMPKYMTWDEFFKKGYFVVPDNPNRKKTVALRWFAEGREKDTPDWGPRLNNQVCRKGLQTTTGKVEFIATSLKNFEEQGYIDEHRPSMHTYVPAWESQKHSPLAVKYPLGMLSPHPRFSMHTMGDGKNSYMNYIKDHRVEVDGYKYWIMRVNSIDAEARGIKNGDLIRAYNDRGSVILAAQVTECLQPGTVHSYESCAVYDPLGTAGKSADRGGCINILTPDRYISKYACGMANNTALVEIEKWDGDKYEIY"
# binding_label = "..................Y....K.IRMTP.D..............RG.....P.K..IAP..AG.K...Y...........................................................................................................................................M........E........F..NI..Q.......D.V.....M....RL...KWFS..................K.........................KTDG.....E..EE....P.CE.RA..RQ...........................................................E.......................................................E..MG..........................................................................................................................................................................................NPNRKK...................................................G.I....................PL.VK..........RF.........N.YM.YIKD............................D.................TECLQP.................................DR...................K.D.....I."
# samples = sample_residue_pos(seq, binding_label, n_samples=5, sample_type="nonbinding_region")

def get_protid_pos_list(id, positions):
    return [(id, pos) for pos in positions]


def plot_dist_mat_by_grouping_axises(vectors, xy_ticks=[], minor_tick_locs=[], labels=[]):
    dist_mat = pairwise_distances(vectors, metric="cosine")

    fig1, ax1 = plt.subplots(1)
    ax1.imshow(dist_mat, cmap='hot', interpolation='bicubic')
    ax1.set_xticks(xy_ticks)
    ax1.xaxis.set_major_formatter(ticker.NullFormatter())
    ax1.xaxis.set_minor_locator(ticker.FixedLocator(minor_tick_locs))
    ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))

    ax1.set_yticks(xy_ticks)
    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1.yaxis.set_minor_locator(ticker.FixedLocator(minor_tick_locs))
    ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(labels))

    plt.show()

def read_data_and_sample():
    f = open("data/downloads/Dataset448.txt", "r")
    seq_start_flag = False
    next_line_label = ""
    seq_count = 0
    seq_reps = {}
    residue_agnostic_samples = {"any": [], "non": [], "dna": [], "rna": [], "prot": [], "smallligand": []}
    seq_embedding = SequenceEmbeddingLayer(embed_format="esm")

    for i, line in enumerate(f.readlines()):
        line = line.rstrip()

        if line.startswith(">"):
            seq_start_flag = True
            id = line[1:]
            next_line_label = "start"
            continue

        if not seq_start_flag: continue
        
        if next_line_label=="start":
            seq = line
            next_line_label = "any"
        elif next_line_label == "any":
            all_binding_sites = line
            next_line_label = "DNA"
        elif next_line_label=="DNA":
            dna_binding_sites = line
            next_line_label = "RNA"
        elif next_line_label=="RNA":
            rna_binding_sites = line
            next_line_label = "prot"
        elif next_line_label == "prot":
            prot_binding_sites = line
            next_line_label = "smallligand"
        elif next_line_label == "smallligand":
            smallligand_binding_sites = line
            next_line_label = "end"
            seq_count += 1

        if next_line_label == "end":
            seq_reps[id] = seq_embedding(id, seq, requires_grad=False)
            # print(f"{seq_count}", f"id: {id}", f"seq: {seq}", f"dna_binding_sites: {dna_binding_sites}", f"rna_binding_sites: {rna_binding_sites}", f"prot_binding_sites: {prot_binding_sites}", f"small_ligand_binding_sites: {small_ligand_binding_sites}", sep="\n")

            sampled_any_binding_pos = sample_residue_binding_pos(seq, all_binding_sites, n_samples=2, sample_type="binding_pos")
            sampled_nonbinding_pos = sample_residue_binding_pos(seq, all_binding_sites, n_samples=2, sample_type="nonbinding_region") # outside binding regions
            sampled_dna_binding_pos = sample_residue_binding_pos(seq, dna_binding_sites, n_samples="all", sample_type="binding_pos")
            sampled_rna_binding_pos = sample_residue_binding_pos(seq, rna_binding_sites, n_samples="all", sample_type="binding_pos")
            sampled_prot_binding_pos = sample_residue_binding_pos(seq, prot_binding_sites, n_samples=2, sample_type="binding_pos")
            sampled_smallligand_binding_pos = sample_residue_binding_pos(seq, smallligand_binding_sites, n_samples=2, sample_type="binding_pos")

            
            residue_agnostic_samples["any"] += get_protid_pos_list(id, sampled_any_binding_pos)
            residue_agnostic_samples["non"] += get_protid_pos_list(id, sampled_nonbinding_pos)
            residue_agnostic_samples["dna"] += get_protid_pos_list(id, sampled_dna_binding_pos)
            residue_agnostic_samples["rna"] += get_protid_pos_list(id, sampled_rna_binding_pos)
            residue_agnostic_samples["prot"] += get_protid_pos_list(id, sampled_prot_binding_pos)
            residue_agnostic_samples["smallligand"] += get_protid_pos_list(id, sampled_smallligand_binding_pos)
            # if len(sampled_dna_binding_pos) > 0:
            #     residue_agnostic_samples["dna"].append((id, sampled_dna_binding_pos))
            # if len(sampled_rna_binding_pos) > 0:
            #     residue_agnostic_samples["rna"].append((id, sampled_rna_binding_pos))
            # if len(sampled_prot_binding_pos) > 0:
            #     residue_agnostic_samples["prot"].append((id, sampled_prot_binding_pos))
            # if len(sampled_small_ligand_binding_pos) > 0:
            #     residue_agnostic_samples["smallligand"].append((id, sampled_small_ligand_binding_pos))


        # if seq_count>=10: break
    return residue_agnostic_samples, seq_reps


# saved_samples_path = "data/tmp/samples.pkl"
# saved_seq_reps_path = "data/tmp/seq_reps.pkl"
# if osp.exists(saved_samples_path):
#     residue_agnostic_samples = Utils.load_pickle(saved_samples_path)
#     seq_reps = Utils.load_pickle(saved_seq_reps_path)
# else: 
#     residue_agnostic_samples, seq_reps = read_data_and_sample()
#     Utils.save_as_pickle(residue_agnostic_samples, saved_samples_path)
#     Utils.save_as_pickle(seq_reps, saved_seq_reps_path)
# # print(residue_agnostic_samples)
# print("Sampling residues done.")


# vectors = []
# for (id, pos) in residue_agnostic_samples["any"]+residue_agnostic_samples["non"]:
#     print(id, pos)
#     vectors.append(seq_reps[id][pos].numpy())
    # break
# print("Generating embeddings of residues done.")
# n_any, n_non = len(residue_agnostic_samples["any"]), len(residue_agnostic_samples["non"])
# plot_dist_mat_by_grouping_axises(vectors, xy_ticks=[n_any], minor_tick_locs=[n_any/2, (n_any+n_any+n_non)/2], labels=['Binding', 'Non-Binding'])

# vectors = []
# for (id, pos) in residue_agnostic_samples["non"]+residue_agnostic_samples["dna"]+residue_agnostic_samples["rna"]+residue_agnostic_samples["prot"]+residue_agnostic_samples["smallligand"]:
#     # print(id, pos)
#     vectors.append(seq_reps[id][pos].numpy())
# print("Generating embeddings of residues done.")

# n_non, n_dna, n_rna, n_prot, n_smallligand = len(residue_agnostic_samples["non"]), len(residue_agnostic_samples["dna"]), len(residue_agnostic_samples["rna"]), len(residue_agnostic_samples["prot"]), len(residue_agnostic_samples["smallligand"]), 
# print(n_non, n_dna, n_rna, n_prot, n_smallligand)
# plot_dist_mat_by_grouping_axises(vectors, xy_ticks=[n_non, n_non+n_dna, n_non+n_dna+n_rna, n_non+n_dna+n_rna+n_prot], 
#                                 minor_tick_locs=[n_non/2, n_non+(n_dna/2), n_non+n_dna+(n_rna/2), n_non+n_dna+n_rna+(n_prot/2), n_non+n_dna+n_rna+n_prot+(n_smallligand/2)], 
#                                 labels=['Non-Binding', "DNA", "RNA", "Protein", "Small-ligand"])


from sklearn.decomposition import PCA, KernelPCA
from scipy.special import softmax

for i in range(10):
    print(f"-------{i}------")
    print(" Generating samples...")
    saved_samples_path = f"data/generated/samples_and_reps/{i}_samples.pkl"
    saved_seq_reps_path = f"data/generated/samples_and_reps/{i}_seq_reps.pkl"
    if osp.exists(saved_samples_path):
        residue_agnostic_samples = Utils.load_pickle(saved_samples_path)
        seq_reps = Utils.load_pickle(saved_seq_reps_path)
    else: 
        residue_agnostic_samples, seq_reps = read_data_and_sample()
        Utils.save_as_pickle(residue_agnostic_samples, saved_samples_path)
        Utils.save_as_pickle(seq_reps, saved_seq_reps_path)
    
    
    print(" Generating embedding of residues...")
    vectors = []
    for (id, pos) in residue_agnostic_samples["non"]+residue_agnostic_samples["dna"]+residue_agnostic_samples["rna"]+residue_agnostic_samples["prot"]+residue_agnostic_samples["smallligand"]:
        # print(id, pos)
        vectors.append(seq_reps[id][pos].numpy())
    

    n_any, n_non, n_dna, n_rna, n_prot, n_smallligand = len(residue_agnostic_samples["any"]), len(residue_agnostic_samples["non"]), len(residue_agnostic_samples["dna"]), len(residue_agnostic_samples["rna"]), len(residue_agnostic_samples["prot"]), len(residue_agnostic_samples["smallligand"]), 
    print(" number of samples: ", n_any, n_non, n_dna, n_rna, n_prot, n_smallligand)
    

    # commulative_sum_of_explained_variance_ratios = []
    # commulative_sum_of_eigenvalues = []
    
    pca = PCA()
    pca.fit(vectors)
    if i==0:
        plt.plot(range(768), np.cumsum(pca.explained_variance_ratio_)+random.uniform(.01, .02), c="b", label="CDF of Explained Variance", alpha=0.3)
    else: 
        plt.plot(range(768), np.cumsum(pca.explained_variance_ratio_)+random.uniform(.01, .02), c="b", alpha=0.3,)
    # normal_singularvalues = pca.singular_values_ / np.sum(pca.singular_values_)
    # plt.plot(range(768), np.cumsum(normal_singularvalues), label="S")
    
    
    
    kpca = KernelPCA(n_components=768, remove_zero_eig=False)
    kpca.fit(vectors)
    normal_eigenvalues = kpca.eigenvalues_ / np.sum(kpca.eigenvalues_)
    if i==0:
        plt.plot(range(768), np.cumsum(normal_eigenvalues), c="orange", label="CDF of Normalized Eigenvalues", alpha=0.3)
    else: plt.plot(range(768), np.cumsum(normal_eigenvalues), c="orange", alpha=0.3)
    
plt.axhline(y=0.82, color = 'r', linestyle = '--', alpha=0.5)
plt.axvline(x=256, color = 'r', linestyle = '--', alpha=0.5)
plt.axhline(y=0.9, color = 'g', linestyle = '--', alpha=0.5)
plt.axvline(x=370, color = 'g', linestyle = '--', alpha=0.5)

plt.xticks([0, 256, 370, 768])
plt.yticks([0, .5, .82, .9, 1])
plt.legend()
# plt.show()
plt.savefig("outputs/images/CDF_pca_var_and_kpca_normal_eigs.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)


