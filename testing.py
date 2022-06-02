# from tensorflow import keras
from datetime import datetime
from typing import Optional, Dict, Any

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from preprocessing import smiles_to_maccs, smiles_to_morgan, read_bit, str2mol

'''
loading models
'''
# property: model
model_set = {
    'wavelength': keras.models.load_model(r"checkpoints/mlp_morgan+maccs.ckpt"),
    'xxxxx': None
}

'''
use the training set to define the min-max scalers for input and output
'''
scaler_set = {}
# absorption wavelength
filepath = 'data/22-01-29-morgan-maccs-train.csv'
data = read_bit(filepath)
data_x_df = pd.DataFrame(data.iloc[:, :-1])
data_y_df = pd.DataFrame(data.iloc[:, -1])
aw_min_max_scaler_X = MinMaxScaler()
aw_min_max_scaler_X.fit(data_x_df)  # Min = 293, Max = 1089
aw_min_max_scaler_y = MinMaxScaler()
aw_min_max_scaler_y.fit(data_y_df)

scaler_set['wavelength'] = (aw_min_max_scaler_X, aw_min_max_scaler_y)


def wavelength_prediction(smiles1, smiles2):
    # SMILES to fingerprints: morgan (x bits) + maccs (y bits)
    mol1, mol2 = str2mol(smiles1), str2mol(smiles2)
    if mol1 is None:
        print(f'ERROR: RDkit failed to transform \"{smiles1}\" to a molecule.')
        return f'ERROR: RDkit failed to transform \"{smiles1}\" to a molecule.'
    if mol2 is None:
        print(f'ERROR: RDkit failed to transform \"{smiles2}\" to a molecule.')
        return f'ERROR: RDkit failed to transform \"{smiles2}\" to a molecule.'
    morgan1, morgan2 = smiles_to_morgan(mol1), smiles_to_morgan(mol2)
    maccs1, maccs2 = smiles_to_maccs(mol1), smiles_to_maccs(mol2)
    X = np.concatenate([morgan1, morgan2, maccs1, maccs2], axis=0).reshape(1, -1)
    scaler_set['wavelength'][0].transform(X)
    output = model_set['wavelength'].predict(X)
    return scaler_set['wavelength'][1].inverse_transform(output)


def wavelength_prediction_batch(_file):
    if _file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(_file, header=None)
        except Exception:
            return "Invalid input file, please check its format."
        # try:
        #     df = pd.read_excel(_file, header=None)
        # except Exception:
        #     return "Invalid input file, please check its format."
    sm_mol1s, sm_mol2s = list(df.iloc[:, 0]), list(df.iloc[:, 1])
    mol1s, mol2s = [str2mol(i) for i in sm_mol1s], [str2mol(i) for i in sm_mol2s]
    morgan1s, morgan2s, maccs1s, maccs2s = [], [], [], []
    valid_orN = []
    for _ in range(len(sm_mol1s)):
        if mol1s[_] and mol2s[_]:
            morgan1s.append(smiles_to_morgan(mol1s[_]))
            morgan2s.append(smiles_to_morgan(mol2s[_]))
            maccs1s.append(smiles_to_maccs(mol1s[_]))
            maccs2s.append(smiles_to_maccs(mol2s[_]))
            valid_orN.append(1)
        else:
            valid_orN.append(0)
    morgan1s = np.array(morgan1s)
    morgan2s = np.array(morgan2s)
    maccs1s = np.array(maccs1s)
    maccs2s = np.array(maccs2s)
    # print(morgan1s.shape, morgan2s.shape, maccs1s.shape, maccs2s.shape)
    X = np.concatenate([morgan1s, morgan2s, maccs1s, maccs2s], axis=1)
    scaler_set['wavelength'][0].transform(X)
    preds = scaler_set['wavelength'][1].inverse_transform(model_set['wavelength'].predict(X)).tolist()
    pred_res = []
    j = 0
    for i in range(len(sm_mol1s)):
        if valid_orN[i] == 1:
            pred_res.append(preds[j][0])
            j += 1
        else:
            pred_res.append('NA')
    # mols, sols, preds, cnt_all, cnt_valid
    return sm_mol1s, sm_mol2s, pred_res


def generate_download_headers(extension: str, filename: Optional[str] = None) -> Dict[str, Any]:
    filename = filename if filename else datetime.now().strftime("%Y%m%d_%H%M%S")
    content_disp = f"attachment; filename={filename}.{extension}"
    headers = {"Content-Disposition": content_disp}
    return headers
