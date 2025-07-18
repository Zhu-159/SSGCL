import argparse
from train_eval import *
from models import *
from util_functions import load_k_fold, MyDynamicDataset

parser = argparse.ArgumentParser(description='SSGCL')

parser.add_argument('--data-name', default='Gdataset',
                    help='dataset name')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--hop', type=int, default=2,
                    help='the number of neighbor (default: 2)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--lambda_reg', type=float, default=0.0001, help='Regularization strength for FCR')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size during training')
parser.add_argument('--dropout_n', type=float, default=0.4,
                    help='random drops neural node with this prob')
parser.add_argument('--dropout_e', type=float, default=0.1,
                    help='random drops edge with this prob')
parser.add_argument('--valid_interval', type=int, default=1)

parser.add_argument('--force-undirected', action='store_true', default=False,
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)



def extract_subgraph(split_data):
    if args.data_name == 'Gdataset':
        print("Using Gdataset with 10% testing...")
        (
            adj_train, train_labels, train_u_indices, train_v_indices,
            test_labels, test_u_indices, test_v_indices
        ) = split_data

    elif args.data_name == 'Cdataset':

        print("Using Cdataset with 10% testing...")
        (
            adj_train, train_labels, train_u_indices, train_v_indices,
            test_labels, test_u_indices, test_v_indices
        ) = split_data
    else:
        print("Using LRSSL with 10% testing...")
        (
            adj_train, train_labels, train_u_indices, train_v_indices,
            test_labels, test_u_indices, test_v_indices
        ) = split_data

    val_test_appendix = str(k) + '_kfold'
    data_combo = (args.data_name, val_test_appendix)

    train_indices = (train_u_indices, train_v_indices)
    test_indices = (test_u_indices, test_v_indices)

    train_file_path = 'data/{}/{}/train'.format(*data_combo)
    train_graph = MyDynamicDataset(train_file_path, adj_train, train_indices, train_labels, args.hop)

    test_file_path = 'data/{}/{}/test'.format(*data_combo)
    test_graph = MyDynamicDataset(test_file_path, adj_train, test_indices, test_labels, args.hop)

    return train_graph, test_graph


if __name__ == '__main__':
    seeds = [20, 34, 42, 43, 61, 70, 83, 1024, 2014, 2047]
    auc_lists, aupr_lists = [], []
    precision_lists, recall_lists, f1_lists = [], [], []
    print(args)
    for seed in seeds:
        print("============= seed=", str(seed), "==================")
        split_data_dict = load_k_fold(args.data_name, seed)

        for k in range(0, 10):
            print('------------ fold', str(k+1), '--------------')
            train_graphs, test_graphs = extract_subgraph(split_data_dict[k])
            model = SSGCL(
                train_graphs,
                latent_dim=[256, 128, 64],
                k=0.6,
                dropout_n=args.dropout_n,
                dropout_e=args.dropout_e,
                force_undirected=args.force_undirected
            )

            print('Used #train graphs: %d, #test graphs: %d' % (
                len(train_graphs),
                len(test_graphs),
            ))

            results = train_epochs(train_graphs, test_graphs, model, args)
            auc_lists.append(results['auroc'])
            aupr_lists.append(results['aupr'])
            precision_lists.append(results['precision'])
            recall_lists.append(results['recall'])
            f1_lists.append(results['f1'])

    print("auroc_list", auc_lists)
    print("aupr_list", aupr_lists)
    print("precision_list", precision_lists)
    print("recall_list", recall_lists)
    print("f1_list", f1_lists)
    print("average auc", np.mean(auc_lists), "average aupr", np.mean(aupr_lists))
    print("average precision", np.mean(precision_lists), "average recall", np.mean(recall_lists), "average f1", np.mean(f1_lists))