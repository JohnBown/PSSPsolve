import json
import numpy as np
import numpy.matlib


def sequence_to_one_hot(sequence=None):
    amino = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
             'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
             'Y': 19, 'V': 20, '-': 0}
    amino_one_hot = [[0] * 20] + np.matlib.eye(20, dtype=int).tolist()

    one_hot = []
    for i in sequence:
        if i in amino.keys():
            one_hot.append(amino_one_hot[amino[i]])
        else:
            one_hot.append(amino_one_hot[amino['-']])

    return one_hot


def trans(pssm):
    return list(map(list, zip(*pssm)))


def expand(body=None, w_len=None, ele=None):
    len = int((w_len - 1) / 2)
    return ele * len + body + ele * len


def slide_raw_data(filepath=None, w_len=25):
    sliding_data = []

    with open(filepath, 'r') as f:
        for data in json.load(f):
            seq_len = data['sequence'].__len__()
            expand_seq = expand(body=data['sequence'], w_len=w_len, ele='-')

            pssm = trans(data['pssm'])
            pssm_e = []

            for l in pssm:
                pssm_e.append(expand(body=l, w_len=w_len, ele=[0]))

            for i in range(seq_len):
                pssm_p = []
                for l in pssm_e:
                    pssm_p.append(l[i:i + w_len])

                one_hot = sequence_to_one_hot(expand_seq[i:i + w_len])

                sliding_data.append({
                    'sequence': expand_seq[i:i + w_len],
                    'input': pssm_p + trans(one_hot),
                    'output': data['secstr'][i]
                })

    return sliding_data


# sliding_datas = slide_raw_data(filepath='json/ss3.json',
#                                w_len=25)
# sliding_datas.__len__()
#
# with open("temp.json", "w") as f:
#     json.dump(sliding_datas[:2000], f)
#
# with open('temp.json','r') as f:
#     temp  = json.load(f)
#     print(temp[10])
