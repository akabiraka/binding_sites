from Bio import SeqIO

for record in SeqIO.parse("data/y.fasta", "fasta"):
    items = record.description.split("|")#[-1].split("\t"))
    pdb_chain_id, uniprot_id = items[0].split(" ")
    tab_sep_DNAorRNA_binding_sites = [] if len(items[1].split("\t")[0])==0 else items[1].split("\t")
    tab_sep_peptide_binding_sites = [] if len(items[2].split("\t")[0])==0 else items[2].split("\t")
    tab_sep_metal_binding_sites = [] if len(items[3].split("\t")[0])==0 else items[3].split("\t")
    tab_sep_regular_binding_sites = [] if len(items[4].split("\t")[0])==0 else items[4].split("\t")

    print(pdb_chain_id, uniprot_id, tab_sep_DNAorRNA_binding_sites, tab_sep_peptide_binding_sites, tab_sep_metal_binding_sites, tab_sep_regular_binding_sites, sep="\n")
