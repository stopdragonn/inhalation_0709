import sys
sys.path.append('../')

import warnings
from rdkit import RDLogger 

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from module.argument import get_parser
from module.read_data import (
    load_data,
    load_pred_data,
    multiclass2binary
)
from module.smiles2fing import smiles2fing
from module.get_model import load_model
from module.common import (
    load_val_result,
    print_best_param,
    calculate_ad_threshold,
    euclidean_distance
)

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

def main():
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    x, y = load_data(path='../data', tg_num=args.tg_num, inhale_type=args.inhale_type)
    y = multiclass2binary(y, args.tg_num)
    
    fingerprints, pred_df, pred_df_origin = load_pred_data()
    
    if (args.tg_num == 403) & (args.inhale_type == 'vapour'):
        args.model = 'lgb'
    elif (args.tg_num == 403) & (args.inhale_type == 'aerosol'):
        args.model = 'rf'
    elif (args.tg_num == 412) & (args.inhale_type == 'vapour'):
        args.model = 'mlp'
    elif (args.tg_num == 412) & (args.inhale_type == 'aerosol'):
        args.model = 'qda'
        args.smoteseed = 0
    elif (args.tg_num == 413) & (args.inhale_type == 'vapour'):
        args.model = 'lda'
        args.smoteseed = 119
    elif (args.tg_num == 413) & (args.inhale_type == 'aerosol'):
        args.model = 'mlp'
    
    val_result = load_val_result(path='..', args=args, is_smote=True)
    best_param = print_best_param(val_result, args.metric)
    
    model = load_model(model=args.model, seed=0, param=best_param)
    
    smote = SMOTE(random_state=args.smoteseed, k_neighbors=args.neighbor)
    x, y = smote.fit_resample(x, y)
    
    model.fit(x, y)
    
    # AD 계산
    D_T = calculate_ad_threshold(x, k=args.ad_k, Z=args.ad_z)
    print(f"Applicability Domain threshold (D_T): {D_T}")

    if args.model == 'plsda':
        pred_score = model.predict(fingerprints)
    else:
        pred_score = model.predict_proba(fingerprints)[:, 1]
    
    # AD 내 여부 확인
    reliability_status = []
    reliability_distance = []
    for fp in fingerprints:
        fp = np.array(fp).reshape(-1)  # Ensure fp is a 1-D array
        ed = euclidean_distance(fp, x)
        reliability_status.append('reliable' if ed < D_T else 'unreliable')
        reliability_distance.append(ed)
    
    pred_df['pred'] = pred_score
    pred_df['reliability_status'] = reliability_status
    pred_df['reliability_distance'] = reliability_distance
    result = pd.merge(pred_df_origin, pred_df[['PREFERRED_NAME', 'SMILES', 'pred', 'reliability_status', 'reliability_distance']], how='left', on=('PREFERRED_NAME', 'SMILES'))
    result.to_excel(f'pred_result/tg{args.tg_num}_{args.inhale_type}_{args.model}.xlsx', header=True, index=False)

if __name__ == '__main__':
    main()
