import os,sys
from Bio import SeqIO
def process_input_faa(input_faa):
    new_input = open(f'Processed_{input_faa}','w')
    for record in SeqIO.parse(input_faa,'fasta').records:
        ID=str(record.id)
        seq=str(record.seq).replace('.','').replace('*','')
        new_input.write(f">{ID}\n{seq}\n")
    new_input.close()

def grouping(b6_file,input_faa):
    quer_lis=[]
    dict={}
    for line in open('OGs_list.txt','r').readlines():
        og=line.strip()
        dict[og]=0
    for line in open(b6_file,'r').readlines():
        temp=line.strip().split('\t')
        querid=temp[0]
        if querid not in quer_lis:
            quer_lis.append(querid)
            OGid=temp[1].split('@')[0]
            dict[OGid]+=1
    out=open(f"input_OG_group.tsv",'w')
    out.write(f'Orthogroup\t{input_faa}\n')
    for o in dict:
        out.write(f'{o}\t{dict[o]}\n')
    out.close()

#1 process input faa
input_faa=sys.argv[1]
process_input_faa(input_faa)
os.system(f"diamond blastp --db  335Fusarium_proteins_OGs_data_db  --query   Processed_{input_faa}  --out  Processed_{input_faa}_vs_db.b6 --outfmt 6 qseqid sseqid pident  evalue bitscore --evalue 1e-10  --max-hsps  1  --max-target-seqs 10")
b6_file=f"Processed_{input_faa}_vs_db.b6"
grouping(b6_file,input_faa)

