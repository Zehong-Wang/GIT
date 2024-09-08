from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from utils.utils import mask2idx, idx2mask


# Helper function

def clean_data(data):
    # Remove list type
    for k in data.keys():
        if isinstance(data[k], list):
            data[k] = None
    return data


def get_pt_loader(data, train_nodes, params):
    return NeighborLoader(
        data, input_nodes=train_nodes, num_neighbors=[params['fanout']] * params["num_layers"],
        batch_size=params["bs"], shuffle=True
    )


def get_sft_loader(data, params):
    task = params["task"]

    if task == "node":
        loader = NeighborLoader(
            data,
            num_neighbors=[10] * params["num_layers"],
            batch_size=params['bs'],
            num_workers=8,
            shuffle=True,
        )
        return loader

    elif task == "edge":
        loader = LinkNeighborLoader(
            data,
            num_neighbors=[10] * params["num_layers"],
            edge_label_index=data.edge_index,
            edge_label=data.y,
            batch_size=params['bs'],
            num_workers=8,
            shuffle=False,
        )
        return loader

    elif task == "graph":
        loader = DataLoader(
            data,
            batch_size=params["bs"],
            shuffle=True,
            num_workers=1,
        )

        return loader


def get_ft_loader(data, split, params):
    task = params["task"]
    setting = params['setting']

    if task == "node":
        data = clean_data(data)
        if setting in ['base', 'few_shot']:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10] * params["num_layers"],
                input_nodes=mask2idx(split["train"]),
                batch_size=params["bs"],
                num_workers=8,
                shuffle=True,
            )
        elif setting in ['in_context', 'zero_shot']:
            train_loader = None
        else:
            raise ValueError('The setting is not supported.')

        val_loader = NeighborLoader(
            data,
            num_neighbors=[-1] * params["num_layers"],
            batch_size=10000,
            num_workers=8,
            shuffle=False,
        )
        test_loader = val_loader

        return train_loader, val_loader, test_loader

    elif task == "edge":
        data = clean_data(data)
        labels = data.y.squeeze()
        if setting in ['base', 'few_shot']:
            train_loader = LinkNeighborLoader(
                data,
                num_neighbors=[30] * params["num_layers"],
                edge_label_index=data.edge_index[:, split["train"]],
                edge_label=labels[split["train"]],
                batch_size=params["bs"],
                num_workers=8,
                shuffle=True,
            )
        elif setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            raise ValueError('The setting is not supported.')

        val_loader = LinkNeighborLoader(
            data,
            num_neighbors=[-1] * params["num_layers"],
            edge_label_index=data.edge_index,
            edge_label=labels,
            batch_size=10000,
            num_workers=8,
            shuffle=False,
        )

        test_loader = val_loader

        return train_loader, val_loader, test_loader

    elif task == "graph":
        if setting in ['base', 'few_shot']:
            train_dataset = data[split["train"]]
            train_loader = DataLoader(
                train_dataset,
                batch_size=params["bs"],
                shuffle=True,
                num_workers=1,
            )
        elif setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            raise ValueError('The setting is not supported.')

        if setting in ['base']:
            val_dataset = data[split["val"]]
            test_dataset = data[split["test"]]

            val_loader = DataLoader(
                val_dataset,
                batch_size=10000,
                shuffle=False,
                num_workers=1,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=10000,
                shuffle=False,
                num_workers=1,
            )

        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            val_loader = DataLoader(
                data,
                batch_size=params["bs"],
                shuffle=True,
                num_workers=1,
            )
            test_loader = val_loader
        else:
            raise ValueError('The setting is not supported.')

        return train_loader, val_loader, test_loader
