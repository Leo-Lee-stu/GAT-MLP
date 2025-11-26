import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import networkx as nx


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
CLQ_DIR = os.path.join(BASE_DIR, '..', 'clq_files')
EXCEL_DIR = os.path.join(BASE_DIR, '..', 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'GNN_models_v2')
CACHED_DIR = os.path.join(BASE_DIR, 'cached_graphs_v2')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHED_DIR, exist_ok=True)

global_feature_scaler = StandardScaler()


def parse_clq_file_v2(file_name):
    cache_path = os.path.join(CACHED_DIR, f"{file_name}.pt")


    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, weights_only=False)
            if data.x.shape[1] == 2:
                return data
        except:
            pass


    file_path = os.path.join(CLQ_DIR, f"{file_name}.clq")
    edges_raw = []
    node_ids = set()

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('e'):
                _, u, v = line.strip().split()
                u, v = int(u), int(v)
                edges_raw.append((u,v))
                node_ids.add(u); node_ids.add(v)

    if not edges_raw:
        return None


    sorted_nodes = sorted(node_ids)
    node_id_map = {old: i for i, old in enumerate(sorted_nodes)}
    num_nodes = len(node_id_map)

    edges = [(node_id_map[u], node_id_map[v]) for u,v in edges_raw]
    edge_index = torch.tensor(edges + [(v,u) for u,v in edges], dtype=torch.long).t().contiguous()


    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    deg = np.array([val for _, val in G.degree()])
    core = np.array([val for _, val in nx.core_number(G).items()])

    features = np.vstack([deg, core]).T
    features = MinMaxScaler().fit_transform(features)
    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    torch.save(data, cache_path)
    return data


def build_label_dataset_v2(label_column, excel_file="method2_dataset_class.xlsx"):
    excel_path = os.path.join(EXCEL_DIR, excel_file)
    df = pd.read_excel(excel_path)

    feature_cols = ['V','E','dmax','davg','D','r','T','Tavg','Tmax','Kavg','k','K']
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

    # 建立标签映射
    label_set = sorted(set(df[label_column]))
    label_map = {label: idx for idx, label in enumerate(label_set)}

    dataset = []
    for _, row in df.iterrows():
        name = row["Graph Name"]
        graph_data = parse_clq_file_v2(name)
        if graph_data is None:
            continue

        graph_data.y = torch.tensor([label_map[row[label_column]]], dtype=torch.long)
        graph_data.global_feats = torch.tensor(row[feature_cols].values.astype(np.float32))
        dataset.append(graph_data)

    return dataset, label_map


class GATWithGlobal_v2(torch.nn.Module):
    def __init__(self, node_in_dim, hidden_dim, global_in_dim, out_dim, dropout=0.5, heads=4):
        super().__init__()
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.fc = torch.nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, global_feats):
        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)


        batch_size = global_feats.size(0)
        global_feats = global_feats.view(batch_size, -1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        x = global_mean_pool(x, batch)
        global_emb = self.global_mlp(global_feats)

        z = torch.cat([x, global_emb], dim=1)
        return self.fc(z)


def train_model_v2(model, dataset, label_map, params):
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    patience = params["patience"]
    dropout = params["dropout"]
    seed = params["seed"]
    test_size = params["test_size"]


    train_data, val_test = train_test_split(dataset, test_size=test_size, random_state=seed,
                                            stratify=[d.y.item() for d in dataset])
    val_data, test_data = train_test_split(val_test, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    all_labels = [d.y.item() for d in dataset]
    class_weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    best_val = 0
    patience_cnt = 0
    best_path = os.path.join(MODEL_DIR, "best_model_v2.pt")


    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            # reshape global feats
            global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, global_feats)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
                out = model(batch.x, batch.edge_index, batch.batch, global_feats)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(batch.y.cpu().numpy().squeeze())

        val_acc = accuracy_score(labels, preds)

        if val_acc > best_val:
            best_val = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    return best_path, test_loader, label_map


def run_once_for_batch_v2(seed=0, test_size=0.2):
    params = {
        "epochs": 50,
        "batch_size": 16,
        "lr": 0.001,
        "hidden_dim": 32,
        "dropout": 0.5,
        "patience": 10,
        "seed": seed,
        "test_size": test_size,
    }


    dataset, label_map = build_label_dataset_v2("class")
    if not dataset:
        return {}

    model = GATWithGlobal_v2(
        node_in_dim=2,
        hidden_dim=params["hidden_dim"],
        global_in_dim=12,
        out_dim=len(label_map),
        dropout=params["dropout"]
    )


    best_model_path, test_loader, label_map = train_model_v2(model, dataset, label_map, params)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = GATWithGlobal_v2(
        node_in_dim=2,
        hidden_dim=params["hidden_dim"],
        global_in_dim=12,
        out_dim=len(label_map),
        dropout=params["dropout"]
    ).to(device)
    eval_model.load_state_dict(torch.load(best_model_path))
    eval_model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            global_feats = torch.stack([d.global_feats for d in batch.to_data_list()]).to(device)
            out = eval_model(batch.x, batch.edge_index, batch.batch, global_feats)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(batch.y.cpu().numpy().squeeze())

    acc = accuracy_score(trues, preds)
    macro = f1_score(trues, preds, average="macro")
    weighted = f1_score(trues, preds, average="weighted")

    return {
        "acc": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "train_time_sec": 0,
        "infer_time_sec": 0,
    }
