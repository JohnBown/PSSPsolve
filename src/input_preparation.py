import os
import gzip
import json
import time
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from Bio import SeqIO
from tqdm import tqdm


# 此模块主要为输入数据准备工作。针对训练文档'ss.txt'的输入，
# 将相关信息存入Protein类的数组并依次把seq_str应用到psi-
# blast中，获得对应位置特异得分矩阵（PSSM）。最终把整理好的
# 数据存入.json文件中，以便以后使用。


def time_print(str):
    print(time.strftime('%Y-%m-%d %X', time.localtime()) + ' ' + str)


class Protein:
    def __init__(self, name="<unknown name>",
                 sequence="<unknown sequence>",
                 secstr="<unkonwn secstr>",
                 pssm=None):

        if not isinstance(name, str):
            raise TypeError("name argument should be a string")
        if not isinstance(sequence, str):
            raise TypeError("sequence argument should be a string")
        if not isinstance(secstr, str):
            raise TypeError("secstr argument should be a string")
        self.name = name
        self.sequence = sequence

        if pssm is None:
            pssm = []
        elif not isinstance(pssm, list):
            raise TypeError("pssm argument should be a 2-dimensional list")
        self.pssm = pssm


def import_raw_data(filepath):
    # 导入包括蛋白质二级结构的原始数据压缩文件
    # 返回：蛋白质实例的列表

    time_print('Import raw data file:"' + filepath + '"')
    with gzip.open(filepath, 'rt') as f:
        seq = ""
        p = None
        protein = []
        for line in f:
            if not line.startswith('>'):
                seq += line[:-1]
            elif seq != "":
                description = line[1:-1].split(':')
                if description[2] == 'sequence':
                    p.secstr = seq.replace(' ', '-')  # 将二级结构中的空格替换
                    protein.append(p)
                else:
                    p = Protein(name=description[0], sequence=seq)
                seq = ""
    time_print('Import completed')
    return protein


def read_pssm(filepath):
    # 从由NCBI psi-blast生成的pssm文件中读取PSSM
    # 返回：位置特异得分矩阵（PSSM）
    pssm = []

    with open(filepath) as f:
        for line in list(f.readlines())[3:-6]:
            pssm.append(list(map(int, line.split()[2:22])))

    return pssm


def seq_to_pssm(name, seq):
    # 输入一条蛋白质序列
    # 返回：位置特异得分矩阵（PSSM）
    seq_fp_tmp = '_seq_temp.fasta'
    pssm_fp_tmp = '_pssm_temp.txt'
    out_fp_tmp = '_out_temp.txt'

    # 手动创建seq的临时序列，以'.fasta'格式呈现
    rec = SeqRecord(Seq(seq, generic_protein),
                    id=name,
                    description="")
    SeqIO.write(rec, seq_fp_tmp, seq_fp_tmp.split(".")[-1])

    # psi-blast搜索生成pssm的文件
    # TODO：需要根据本机配置情况进行修改
    os.system('psiblast '
              '-db ~/Database/pdbaa '
              '-query {0} '
              '-out_ascii_pssm {1} '
              '-out {2} '
              # '-evalue 10 '   # 阈值
              # '-word_size 3 '
              # '-gapopen <int> '
              # '-gapextend <int> '
              # '-matrix BLOSUM62'
              # '-threshold 0.001 '
              '-num_iterations 3 '
              '-comp_based_stats 1 '  # 不添加这条会产生警告
              .format(seq_fp_tmp, pssm_fp_tmp, out_fp_tmp))

    if os.path.exists(pssm_fp_tmp):
        pssm = read_pssm(pssm_fp_tmp)
    else:
        return None

    if os.path.exists(seq_fp_tmp):
        os.remove(seq_fp_tmp)
    if os.path.exists(pssm_fp_tmp):
        os.remove(pssm_fp_tmp)
    if os.path.exists(out_fp_tmp):
        os.remove(out_fp_tmp)

    return pssm


def raw_data_to_json(filepath):
    protein = import_raw_data(filepath)
    time_print('Calculate the PSSM of each protein, total ' + protein.__len__().__str__())
    p_json = []
    count = 5000

    for i, p in enumerate(tqdm(protein, desc="Calculate PSSM")):
        p.pssm = seq_to_pssm(name=p.name, seq=p.sequence)
        if p.pssm is None:
            continue
        # 由于设计疏漏，最后一个蛋白质二级结构少一个字符
        # 为了避免此问题，将缺失数据（1条）直接舍去
        assert len(p.pssm) == len(p.sequence)
        p_json.append(p.__dict__)

        if ((i+1) % count) == 0:
            # Writing JSON data
            with open('json/ss' + str(i // count) + '.json', 'w') as f:
                json.dump(p_json, f)
                p_json = []

    time_print('Import completed')


# os.system('rm -rf json/\n'
#           'mkdir json/\n')
# raw_data_to_json('ss.txt.gz')
