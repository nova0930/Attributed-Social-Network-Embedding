import pandas as pd
import os

from convertdata import load_gold_standard


def biogrid(path, gene_identifier):
    # yeast_id = 559292
    if gene_identifier == 'yeast':
        gene_id = 559292
        file_id = 4
    elif gene_identifier == 'ecoli':
        gene_id = 316407
        file_id = 3
    elif gene_identifier == 'aureus':
        gene_id = 559292
        file_id = 2
    output_file = path + "edgelist_biogrid_name.csv"
    gene_id_file = path + "net"+ str(file_id) + "_gene_ids.tsv"
    gene_list = pd.read_csv(gene_id_file, sep="\t")
    if os.path.isfile(output_file)==False:
        file_name = './data/BIOGRID-ALL-3.4.153.tab2.txt'
        gold_standard_data = pd.read_csv(file_name, sep="\t")

        data = gold_standard_data[(gold_standard_data['Organism Interactor A'] == gene_id) & (gold_standard_data['Organism Interactor B'] == gene_id)]

        if (file_id ==4):
            data = data[['Systematic Name Interactor A', 'Systematic Name Interactor B']]
        elif (file_id ==3):
            data = data[['Official Symbol Interactor A', 'Official Symbol Interactor B']]
        data = data.drop_duplicates(subset=['Official Symbol Interactor A', 'Official Symbol Interactor B'], keep='first')
        data.to_csv(output_file, index=False, header=False)

    gold_standard_data = pd.read_csv(output_file, sep=",")
    gold_standard_data.columns = ['gene','gene2']
    gold_standard_data['relation'] = 1
    gene_list.columns = ['id','gene']

    merged = pd.merge(gene_list, gold_standard_data, on="gene")
    merged.columns = ['id','gene2', 'gene', 'relation']

    merged2 = pd.merge(gene_list, merged, on="gene")

    final_goldStandard = merged2[['id_x','id_y','relation']]
    final_goldStandard = final_goldStandard.sort_values(by=['relation'], ascending=False)
    final_goldStandard = final_goldStandard[final_goldStandard['id_x'] != final_goldStandard['id_y']]
    final_goldStandard.to_csv(path + "biogrid_edgelist.csv", index=False, header=False)


    edgelist_df = pd.DataFrame(load_gold_standard(final_goldStandard)).astype(int)
    edgelist_df.to_csv(path + "edgelist_biogrid.txt", header=None, sep=' ', index=False, mode='a')