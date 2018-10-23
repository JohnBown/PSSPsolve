import json


def trans(pssm):
    return list(map(list, zip(*pssm)))


def expand(body=None, w_len=None, ele=None):
    len = int((w_len - 1) / 2)
    return ele * len + body + ele * len


def slide_raw_data(filepath=None, w_len=25):
    sliding_data = []

    with open(filepath, 'r') as f:
        datas = json.load(f)

        for data in datas:
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

                sliding_data.append({
                    'name': data['name'],
                    'sequence': expand_seq[i:i + w_len],
                    'secstr': data['secstr'][i],
                    'pssm': pssm_p
                })

    return sliding_data


# sliding_data = slide_raw_data(filepath='json/ss3.json', w_len=25)
# print(sliding_data.__len__())
# print(sliding_data[666])
