import csv
from itertools import islice

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

NUM_ATTR = 167


def read_bit(filepath):
    data = list()
    with open(filepath, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        for row in islice(reader, 1, None):
            temp0 = list(row[1])
            temp0 = temp0 + list(row[2])

            temp = row[3].strip().split(' ')
            temp = [int(x) for x in temp]
            bits_1 = [0 for x in range(NUM_ATTR)]
            for t in temp:
                bits_1[t] = 1

            temp = row[4].strip().split(' ')
            temp = [int(x) for x in temp]

            bits_2 = [0 for _ in range(NUM_ATTR)]
            for t in temp:
                bits_2[t] = 1

            bits = bits_1 + bits_2

            temp = bits
            temp.append(float(row[0]))

            temp = temp0 + temp

            data.append(temp)
    data = np.array(data)
    data = pd.DataFrame(data)
    return data


def str2mol(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    return mol


def smiles_to_maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    return [int(i) for i in fp]


def smiles_to_morgan(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()
    return [int(i) for i in fp]

    smile_list = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in islice(reader, 0, None):
            # for row in islice(reader, 1, None):
            if len(row) == 0:
                smile_list.append('-')
                continue
            smile_list.append(Chem.MolFromSmiles(row[0]))
        fps = []
        bi_list = []
        for x in smile_list:
            if x == '-':
                fps.append(['/'])
                bi_list.append(['/'])
                continue
            bi = {}
            temp = object()
            if method == 'topo':
                temp = Chem.RDKFingerprint(x, maxPath=2, bitInfo=bi)
            elif method == 'morgan':
                temp = AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024, bitInfo=bi)
            elif method == 'maccs':
                temp = MACCSkeys.GenMACCSKeys(x)
            fps.append(temp)
            _str = temp.ToBitString()
            if method == 'maccs':
                _temp_str = ''
                print(len(_str))
                for i in range(len(_str)):
                    if _str[i] == '1':
                        _temp_str += (' ' + str(i))
                bi_list.append([_temp_str])
            else:
                bi_list.append([str])
            print(bi_list)

        header = [method + ' fingerprint', 'bit string']
        filepath = 'data/fp/' + dir + csv_file_name.replace('.txt', '') + '_' + method + '_fp_bit.csv'
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(header)
            writer.writerows(bi_list)
