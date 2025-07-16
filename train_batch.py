import argparse
import torch
from model import Model
from main import run_trainval
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data (e.g., train_data_10.pkl)')
    args = parser.parse_args()

    train_data_path = args.train_data
    tag = train_data_path.replace('.pkl', '').replace('/', '_')

    model_save_path = f'model_{tag}.pt'
    edge_save_path = f'edge_importance_{tag}.npy'

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Graph args
    graph_args = {'max_hop': 2, 'num_node': 120}

    # Init model
    model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
    model.to(device)

    # Train and validate
    run_trainval(model, pra_traindata_path=train_data_path, pra_testdata_path='test_data.pkl')

    # Save edge importance
    with torch.no_grad():
        edge_importance = model.edge_importance.cpu().numpy()
        np.save(edge_save_path, edge_importance)
        print(f"Saved edge importance to {edge_save_path}")

    # Save model
    torch.save({'xin_graph_seq2seq_model': model.state_dict()}, model_save_path)
    print(f"Saved model to {model_save_path}")

if __name__ == '__main__':
    main()
