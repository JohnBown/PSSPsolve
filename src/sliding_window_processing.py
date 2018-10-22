import json


def trans(pssm):
    return list(map(list, zip(*pssm)))


def expand(body=None, w_len=None, ele=None):
    len = int((w_len - 1) / 2)
    return ele * len + body + ele * len


def slide_raw_data(filepath=None, w_len=25):
    sliding_data = []

    with open(filepath, 'r') as f:
        data = json.load(f)

        for index in range(data.__len__()):

            seq_len = data[index]['sequence'].__len__()
            expand_seq = expand(body=data[index]['sequence'], w_len=w_len, ele='-')

            pssm = trans(data[index]['pssm'])
            pssm_e = expand(body=pssm, w_len=w_len, ele=[[0] * 20])
            pssm_t = trans(pssm_e)

            for i in range(seq_len):
                pssm_p = []
                for l in pssm_t:
                    pssm_p.append(l[i:i + w_len])

                sliding_data.append({
                    'name': data[index]['name'],
                    'sequence': expand_seq[i:i + w_len],
                    'secstr': data[index]['secstr'][i],
                    'pssm': pssm_p
                })

    return sliding_data


sliding_data = slide_raw_data(filepath='json/ss3.json', w_len=25)
print(sliding_data.__len__())
