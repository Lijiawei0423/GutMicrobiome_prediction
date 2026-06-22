import pandas as pd
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
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

dpath = '/home1/LIJW/AbundanceData/'
result_path = '/home1/LIJW/0129JiangNanResults'

class GutFloraDataset(Dataset):
    def __init__(self,data,tmp_lst,target_name):
        self.data = data
        self.features = tmp_lst
        self.target_name = target_name
        self.feature_data = self.data[self.features].fillna(0).values
        
        'For regression, apply log1p to the target.'
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
        'The transformer hidden size must be divisible by 4.'
        # output_dim = input_dim - (input_dim % 4)
        'Use a fixed hidden size of 100.'
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

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, targets,targets_log in dataloader:
        features, targets ,targets_log = features.to(device), targets.to(device),targets_log.to(device)
        optimizer.zero_grad()
        outputs = model(features.unsqueeze(1))# batch_first=True, so the output layout is (batch, seq, feature)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("Found NaN or Inf in outputs!")
            print(outputs)
            exit()
        if (targets > 0).sum() == 0:
            loss = (outputs.view(-1) - outputs.view(-1).detach()).pow(2).mean()
        else:
            loss = criterion(outputs.view(-1), targets_log.float().view(-1))
            loss = torch.where(targets>0, loss, 0).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)

def evaluate_model (model,dataloader,device):
    model.eval()
    total_corr = 0.0
    valid_corr_count = 0
    
    with torch.no_grad():
        for features,targets,targets_log in dataloader:
            features, targets ,targets_log = features.to(device), targets.to(device) ,targets_log.to(device)
            outputs = model(features.unsqueeze(1))
            if (targets>0).sum().item()>0: # Only evaluate batches that contain target-positive samples.
                outputs_cpu = outputs.view(-1).cpu()
                targets_log_cpu = targets_log.view(-1).cpu()
                'Compute Pearson correlation.'
                corr, _ = pearsonr(outputs_cpu,targets_log_cpu)
                total_corr += corr
                valid_corr_count += 1
    
    avg_corr = total_corr/ valid_corr_count if valid_corr_count>0 else float('nan')
    if dist.get_rank() == 0:
        return avg_corr
    else:
        return 0.0 


def get_top_features(feature_df):
    top_total = feature_df.nlargest(100, 'Total_Gain')['Analyst'].tolist()
    return  top_total

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Model Training')
    parser.add_argument('--clade_name', type=str, required=True, help="Clade name for classification")
    parser.add_argument('--action_type', type=int, required=True, help="Action type (integer value)")
    return parser.parse_args()

def model_train_valid(train_df,test_df,clade_name,tmp_lst,rank,world_size,device,total_epochs,cv_id):
    train_data = GutFloraDataset(train_df, tmp_lst, clade_name)
    valid_data = GutFloraDataset(test_df, tmp_lst, clade_name)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank ,shuffle = True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, num_replicas=world_size, rank=rank,shuffle = False,drop_last = False)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=False, sampler=train_sampler,drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, sampler=valid_sampler,drop_last = False)
    
    model = TransformerModel(input_dim=len(tmp_lst)).to(device)
    model = DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True
            # gradient_as_bucket_view
        )
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_corr = 0.0
    
    best_model_state = None
    best_epoch_preds = None
    best_epoch_targets = None
    best_epoch_eids = None
    
    for epoch in range(total_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_model(model,train_loader,criterion,optimizer,device)
        valid_corr = evaluate_model(model,valid_loader,device)
        if rank==0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Corr: {valid_corr}')
            if epoch == 0:
                best_corr = valid_corr
                best_model_state = model.state_dict()
            else:
                if valid_corr>best_corr:
                    best_corr = valid_corr
                    best_model_state
                    best_model_state = model.state_dict()
                    model.eval()
                    
                    all_preds = []
                    all_targets = []
                    all_targets_log = []
                    all_eids = []
                    
                    with torch.no_grad():
                        for idx,(features, targets, targets_log) in enumerate(valid_loader):
                            features, targets ,targets_log = features.to(device), targets.to(device),targets_log.to(device)
                            outputs = model(features.unsqueeze(1)).view(-1)
                            all_preds.append(outputs.cpu().numpy())
                            all_targets.append(targets.cpu().numpy())
                            all_targets_log.append(targets_log.cpu().numpy())
                            start_idx = idx * valid_loader.batch_size
                            batch_eids = test_df.iloc[start_idx:start_idx + len(targets)]['eid'].tolist()
                            all_eids.extend(batch_eids)
                    best_epoch_preds = all_preds
                    best_epoch_targets = all_targets
                    best_epoch_targets_log = all_targets_log
                    best_epoch_eids = all_eids
                    
                    
    if rank == 0:
        'Save the checkpoint.'
        model_dir = os.path.join(result_path, clade_name, 'S2_Model')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'fold{cv_id}_2_best_model_TotalGain.pth')
        torch.save(best_model_state, model_path)
        print(f'✅ Saved best model to {model_path}')
        'Save validation risk scores only.'
        # risk_df = pd.DataFrame({
        #     'eid': best_epoch_eids,
        #     'risk_score': best_epoch_preds,
        #     'true_label': best_epoch_targets,
        #     'true_label_log': best_epoch_targets_log
        # })
        # pred_dir = os.path.join(result_path, clade_name, 'S3_Pred')
        # risk_path = os.path.join(pred_dir,  f'fold{cv_id}_2_best_riskscore_TotalGain.csv')
        # risk_df.to_csv(risk_path, index=False)
        # print(f'✅ Saved best risk scores to {risk_path}')
        
        print(f'✅ After training, Best Validation Corr: {best_corr}')
    return best_corr


def main(rank,world_size,clade_name):
    deepspeed.init_distributed(
        dist_backend='nccl',
        auto_mpi_discovery=True,
        verbose=False,
        init_method=None,
        distributed_port=29600
    )
    gpu_idx = rank
    print('Using gpu_index:', gpu_idx)
    torch_device = torch.device("cuda", gpu_idx)
    torch.cuda.set_device(torch_device)
    device = torch_device
    
    data_df = pd.read_csv(dpath+'AbundanceData_preprocessed.csv')
    cv_df = pd.read_csv(dpath+'PhenotypeData.csv',usecols=['eid','cv_id'])
    df = pd.merge(data_df, cv_df, how='inner', on=['eid'])
    fold_results = []
    for cv_id in range(5):
        if rank == 0:
            print('-----------------------begin cv_id-------------------:', cv_id)
        train_df = df[df['cv_id'] != cv_id]
        valid_df = df[df['cv_id'] == cv_id]
        'Load the selected feature list.'
        imp_df = pd.read_csv(result_path+'/'+clade_name+'/S1_FS/Importance_2_all.csv')
        top_total = get_top_features(imp_df)
        total_epochs = 30
        # if rank == 0:
        #     print(f'❗️For: {clade_name} Begin: LightGBM')
        # lightgbm_corr = model_train_valid(train_df,valid_df,clade_name,top_lightgbm,rank,world_size,device,total_epochs)
        # if rank == 0:
        #     print(f'❗️For: {clade_name} Begin: XGBoost')
        # xgboost_corr = model_train_valid(train_df,valid_df,clade_name,top_xgboost,rank,world_size,device,total_epochs)
        # if rank == 0:
        #     print(f'❗️For: {clade_name} Begin: CatBoost')
        # catboost_corr = model_train_valid(train_df,valid_df,clade_name,top_catboost,rank,world_size,device,total_epochs)  
        if rank == 0:
            print(f'❗️For: {clade_name} Begin: TotalGain') 
        totalgain_corr = model_train_valid(train_df,valid_df,clade_name,top_total,rank,world_size,device,total_epochs,cv_id)
        
        # Save fold-level results.
        fold_results.append({
            "Analyst": clade_name,
            "cv_id": cv_id,
            # 'LightGBM_Corr': lightgbm_corr,
            # 'XGBoost_Corr': xgboost_corr,
            # 'CatBoost_Corr': catboost_corr,
            'TotalGain_Corr': totalgain_corr
        })
    
    if rank == 0:
        fold_df = pd.DataFrame(fold_results)
        save_per_fold_csv = os.path.join('/home1/LIJW/JiangNan_results2',clade_name,'S4_Eval/2_fold_results_TotalGain.csv')
        fold_df.to_csv(save_per_fold_csv, index=False)
        print(f"Saved per-fold results to {save_per_fold_csv}")
    return 0
    #     avg_lightgbm_corr = np.mean([results[cv_id]['LightGBM_Corr'] for cv_id in results])
    #     avg_xgboost_corr = np.mean([results[cv_id]['XGBoost_Corr'] for cv_id in results])
    #     avg_catboost_corr = np.mean([results[cv_id]['CatBoost_Corr'] for cv_id in results])
    #     avg_totalgain_corr = np.mean([results[cv_id]['TotalGain_Corr'] for cv_id in results])
        
    #     print("Average Correlations across folds:")
    #     print(f"LightGBM_Corr: {avg_lightgbm_corr}")
    #     print(f"XGBoost_Corr: {avg_xgboost_corr}")
    #     print(f"CatBoost_Corr: {avg_catboost_corr}")
    #     print(f"TotalGain_Corr: {avg_totalgain_corr}")
    # return avg_lightgbm_corr,avg_xgboost_corr,avg_catboost_corr,avg_totalgain_corr

if __name__ == '__main__':
    args = parse_args()
    clade_name = args.clade_name
    action_type = args.action_type

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE')
                     )
    main(local_rank, world_size,clade_name)
