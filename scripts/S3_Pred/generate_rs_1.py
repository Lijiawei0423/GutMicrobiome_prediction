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


class GutFloraDataset(Dataset):
    def __init__(self,data,tmp_lst,target_name):
        self.data = data
        self.features = tmp_lst
        self.target_name = target_name
        self.feature_data = self.data[self.features].fillna(0).values

        self.target_data = self.data[self.target_name].fillna(0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.feature_data[idx]
        targets = self.target_data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()
    
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
        pred_zero = torch.sigmoid(self.fc2(x))
        # pred_zero = self.fc2(x)
        return pred_zero
    
class GutFloraDataset(Dataset):
    def __init__(self,data,tmp_lst,target_name):
        self.data = data
        self.features = tmp_lst
        self.target_name = target_name
        self.feature_data = self.data[self.features].fillna(0).values
        self.target_data = self.data[self.target_name].fillna(0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.feature_data[idx]
        targets = self.target_data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).squeeze()
    

def get_top_features(feature_df):
    top_total = feature_df.nlargest(100, 'Total_Gain')['Analyst'].tolist()
    return  top_total

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Model Training')
    parser.add_argument('--clade_name', type=str, required=True, help="Clade name for classification")
    return parser.parse_args()


dpath = '...'
result_path = '...'


args = parse_args()
action_type = 1
clade_name = args.clade_name


data_df = pd.read_csv(dpath+'AbundanceData_preprocessed.csv')
cv_df = pd.read_csv(dpath+'PhenotypeData.csv',usecols=['eid','cv_id'])
df = pd.merge(data_df, cv_df, how='inner', on=['eid'])

imp_df = pd.read_csv(result_path+'/'+clade_name+'/S1_FS/Importance_1_all.csv')
tmp_lst = get_top_features(imp_df)
input_dim = len(get_top_features(imp_df))


model_dir = os.path.join(result_path, clade_name, 'S2_Model')
model_cache = {}

for cv_id in range(5):
    model_path = os.path.join(model_dir, f'fold{cv_id}_{action_type}_best_model_TotalGain.pth')
    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found: {model_path}. Skipping this model.")
        continue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model_cache[cv_id] = (model)
    



for cv_id in range(5):
    if cv_id not in model_cache:
        continue
    model = model_cache[cv_id]
    all_results = []
    valid_df = df[df['cv_id'] == cv_id]
    # tmp_lst = get_top_features(imp_df)
    # device = "cpu"
    
    # model = TransformerModel(input_dim = input_dim).to(device)

    # model_dir = os.path.join(result_path, clade_name, 'S2_Model')
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, f'fold{cv_id}_{action_type}_best_model_TotalGain.pth')
    # if not os.path.exists(model_path):
    #     print(f"[Warning] Model file not found: {model_path}. Skipping this model.")
    #     continue
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    valid_dataset = GutFloraDataset(valid_df, tmp_lst, target_name=clade_name) 
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    eids = valid_df["eid"].tolist()
    eid_idx = 0
    with torch.no_grad():
        for features, targets in valid_loader:
            features,targets = features.to(device),targets.to(device)
            outputs = model(features.unsqueeze(1)).view(-1)
            targets_binary = (targets > 0).int().view(-1)
            batch_size = features.shape[0]
            for i in range(batch_size):
                all_results.append({
                    "eid": eids[eid_idx],
                    "cv_id": cv_id,
                    "risk_score": outputs[i].item(),
                    "target": targets[i].item(),
                    "target_binary": targets_binary[i].item()
                })
                eid_idx += 1
            
    results_df = pd.DataFrame(all_results)
    rs_dir = os.path.join(result_path, clade_name, 'S3_Pred')
    os.makedirs(rs_dir, exist_ok=True)
    rs_path = os.path.join(rs_dir, f'fold{cv_id}_risk_scores.csv')
    results_df.to_csv(rs_path, index=False)
    print(f"Fold {cv_id} risk scores saved to {rs_path}")
    
    
    
    
