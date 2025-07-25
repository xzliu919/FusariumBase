import os
from Bio import SeqIO

faa_file_path='/data/xzliu/01_Fusarium_project/01_300Fusarium_project_analysis/00_356Fusarium_genome_files/faas'
all_AAs={}
for file in os.listdir(faa_file_path):
    species=file.split('.faa')[0]
    for record in SeqIO.parse(f'{faa_file_path}/{file}','fasta').records:
        AAID=str(record.id)
        Seq=str(record.seq).replace('.','').replace('*','')
        ID=f'{species}@{AAID}'
        all_AAs[ID]=Seq
print(f'all number of AAs: {len(all_AAs)}')
new_all_aas=open('335Fusarium_proteins_OGs_data.fasta','w')
for line in open('Orthogroups.tsv','r').readlines()[1:]:
    temp=line.strip().split('\t')
    OGID=temp[0]
    for cods in temp[1:]:
        if len(cods.split(', '))>1:
            for i in cods.split(', '):
                if i in all_AAs:
                    new_all_aas.write(f">{OGID}@{i}\n{all_AAs[i]}\n")
        else:
            if cods in all_AAs:
                new_all_aas.write(f">{OGID}@{cods}\n{all_AAs[cods]}\n")
