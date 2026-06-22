import pandas as pd
import subprocess
from tqdm import tqdm
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import deepspeed
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from collections import OrderedDict
import numpy as np

class GutFloraDataset(Dataset):
    def __init__(self,data,tmp_lst,target_name):
        self.data = data
        self.features = tmp_lst
        self.target_name = target_name
        self.feature_data = self.data[self.features].fillna(0).values
        self.target_data = self.data[self.target_name].fillna(0).values
        self.target_log = np.log1p(self.target_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.feature_data[idx]
        target = self.target_data[idx]
        target_log = self.target_log[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32),torch.tensor(target_log,dtype=torch.float32)
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        output_dim = 100
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=4,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc2 = nn.Linear(output_dim, 1)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        pred_value = self.fc2(x)
        return pred_value


def get_top_features(feature_df):
    top_total = feature_df.nlargest(100, 'Total_Gain')['Analyst'].tolist()
    return  top_total

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Model Training')
    parser.add_argument('--clade_name', type=str, required=True, help="Clade name for classification")
    return parser.parse_args()


dpath = '/home1/LIJW/AbundanceData/'
result_path = '/home1/LIJW/0129JiangNanResults'


args = parse_args()
action_type = 2
clade_name = args.clade_name


data_df = pd.read_csv(dpath+'AbundanceData_preprocessed.csv')
cv_df = pd.read_csv(dpath+'PhenotypeData.csv',usecols=['eid','cv_id'])
df = pd.merge(data_df, cv_df, how='inner', on=['eid'])

for cv_id in range(5):
    all_results = []
    valid_df = df[df['cv_id'] == cv_id]
    imp_df = pd.read_csv(result_path+'/'+clade_name+'/S1_FS/Importance_2_all.csv')
    tmp_lst = get_top_features(imp_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_dim = len(tmp_lst)).to(device)

    model_dir = os.path.join(result_path, clade_name, 'S2_Model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'fold{cv_id}_{action_type}_best_model_TotalGain.pth')
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    valid_dataset = GutFloraDataset(valid_df, tmp_lst, target_name=clade_name)  
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    
    eids = valid_df["eid"].tolist()
    eid_idx = 0
    
    with torch.no_grad():
        for features, targets,targets_log in valid_loader:
            features, targets ,targets_log = features.to(device), targets.to(device),targets_log.to(device)
            outputs = model(features.unsqueeze(1)).view(-1)
            batch_size = features.shape[0]
            
            for i in range(batch_size):
                all_results.append({
                    "eid": eids[eid_idx],
                    "cv_id": cv_id,
                    "target": targets[i].item(),
                    "target_log": targets_log[i].item(),
                    "pred_value": outputs[i].item()
                })
                eid_idx += 1
    results_df = pd.DataFrame(all_results)
    rs_dir = os.path.join(result_path, clade_name, 'S3_Pred')
    os.makedirs(rs_dir, exist_ok=True)
    rs_path = os.path.join(rs_dir, f'fold{cv_id}_risk_scores.csv')
    results_df.to_csv(rs_path, index=False)
    print(f"Fold {cv_id} risk scores saved to {rs_path}")
            
    