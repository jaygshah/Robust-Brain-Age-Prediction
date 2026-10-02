from collections import defaultdict
from data_loaders_4binover import get_loader, list_age_folders
from logger import Logger
import argparse, datetime, json
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F

from metrics import expected_age, mean_absolute_error, ordinality, systematic_bias
from ordinal_loss import order_loss, euclidean_loss, order_loss_p
from mean_variance_loss import MeanVarianceLoss

from models import resnet

# Constants for MeanVarianceLoss
LAMBDA_1 = 0.05
LAMBDA_2 = 0.2

mean_variance_loss = MeanVarianceLoss(LAMBDA_1, LAMBDA_2)

# Feature-space distance regularizers, weighted by --ld
distance_losses = {
    "order": order_loss,
    "euc": euclidean_loss,
    "order_p": order_loss_p,
}

LOSSES = ["ce", "mv", "mse", *distance_losses]
MODEL_DEPTHS = {"resnet10": 10, "resnet18": 18, "resnet34": 34, "resnet50": 50, "resnet101": 101}


# Function to get the number of parameters in a model
def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(model_name, num_classes):
    return resnet.generate_model(model_depth=MODEL_DEPTHS[model_name],
                                 n_classes=num_classes,
                                 n_input_channels=1,
                                 widen_factor=1)


def compute_losses(features, y_, y, args):
    target = y[:, 0]
    losses = {}
    for loss in args.losses:
        if loss in distance_losses:
            kwargs = {"p": args.p} if loss == "order_p" else {}
            losses[loss] = args.ld * distance_losses[loss](features, y, max_label_diff=args.max_label_diff, **kwargs)
        elif loss == "mv":
            mean_loss, var_loss = mean_variance_loss(y_, target)
            losses[loss] = mean_loss + var_loss
        elif loss == "mse":
            losses[loss] = F.mse_loss(y_[:, 0], target.float())
        else:
            losses[loss] = F.cross_entropy(y_, target)
    return losses


def predict_age(y_, args):
    if args.regression:
        return y_[:, 0] + args.min_age
    return expected_age(y_, args.min_age)


# Training function
def train(model, loader, args, opt, epoch, device, logger):

    model.train()
    metrics = defaultdict(list)

    for it, (x, y, _) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        features, y_ = model(x)
        losses = compute_losses(features, y_, y, args)
        closs = sum(losses.values())

        opt.zero_grad()
        closs.backward()
        opt.step()

        step = epoch * len(loader) + it + 1
        for name, value in [*losses.items(), ("total_loss", closs)]:
            metrics[name].append(value.item())
            logger.scalar_summary("train/" + name, value.item(), step)

        log = f"Epoch {epoch+1}, Iter {it+1}/{len(loader)}:"
        for m in sorted(metrics.keys()):
            log += f" {m} = {np.mean(metrics[m]):.4f}"
        print(log)

    return metrics


# Validation and testing function
@torch.no_grad()
def evaluate(model, loader, args, device, compute_loss=True):
    """Return (metrics, per-subject predictions). Ages are in years; BrainAGE = predicted - true age."""

    model.eval()
    loss_values = defaultdict(list)
    features_all, true_ages, pred_ages, pred_classes, paths = [], [], [], [], []

    for x, y, url in loader:
        x = x.to(device)
        y = y.to(device)

        features, y_ = model(x)

        if compute_loss:
            losses = compute_losses(features, y_, y, args)
            for name, value in losses.items():
                loss_values[name].append(value.item())
            loss_values["total_loss"].append(sum(losses.values()).item())

        features_all.append(features.cpu())
        true_ages.append((y[:, 0] + args.min_age).cpu())
        pred_ages.append(predict_age(y_, args).cpu())
        if not args.regression:
            pred_classes.append((torch.argmax(y_, dim=1) + args.min_age).cpu())
        paths += list(url)

    true_ages = torch.cat(true_ages).numpy()
    pred_ages = torch.cat(pred_ages).numpy()

    metrics = {name: float(np.mean(values)) for name, values in loss_values.items()}
    metrics["mae"] = mean_absolute_error(pred_ages, true_ages)
    metrics["sb_l"], metrics["sb_r"] = systematic_bias(pred_ages, true_ages)
    metrics["ordinality"] = ordinality(torch.cat(features_all).numpy(), true_ages)

    predictions = {"path": paths, "age": true_ages, "predicted_age": pred_ages, "brainage": pred_ages - true_ages}
    if not args.regression:
        pred_classes = torch.cat(pred_classes).numpy()
        metrics["accuracy"] = float(np.mean(pred_classes == true_ages))
        predictions["predicted_class"] = pred_classes

    return metrics, pd.DataFrame(predictions)


def format_metrics(metrics):
    return " ".join(f"{k} = {v:.4f}" for k, v in metrics.items())


# Main function to parse arguments and run training and testing
def main():

    parser = argparse.ArgumentParser(description='3D Brain Age prediction')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='number of samples in each batch')
    parser.add_argument('--patience', default=0, type=int, metavar='N',
                        help='stop after N epochs without val MAE improvement (0 disables early stopping)')
    parser.add_argument('--seed', default=3141, type=int)

    # Data Parameters
    parser.add_argument('-d', '--dataset', default='data', type=str,
                        help='folder containing HC/{train,val,test}/<age>/*.nii (see age_binning.py)')
    parser.add_argument('--min_age', default=None, type=int,
                        help='youngest age class (default: youngest age folder in the training set)')
    parser.add_argument('--max_age', default=None, type=int,
                        help='oldest age class (default: oldest age folder in the training set)')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='disable training-time data augmentation')
    parser.add_argument('--num_workers', default=16, type=int)

    # Losses used
    parser.add_argument('--losses', default=['ce'], type=str, nargs='+', choices=LOSSES,
                        help='losses (default: Cross Entropy (ce), add others)')
    parser.add_argument('--ld', default=0.1, type=float, metavar='N',
                        help='weight of the distance loss (order, euc, order_p)')
    parser.add_argument('--order_weights', default='batch', choices=['batch', 'age'],
                        help="pair weights of the distance loss: 'batch' = per-batch (w - min) / max scaling used "
                             "for the WACV 2024 results; 'age' = |age difference| / (max_age - min_age)")
    parser.add_argument('--p', default=0.5, type=float,
                        help='Lk distance used by order_p')

    # Model Parameters
    parser.add_argument('--model_name', default="resnet18", type=str, choices=list(MODEL_DEPTHS),
                        help='name of the classification model')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, choices=['adamw', 'adam'],
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)

    # Outputs
    parser.add_argument('--output_dir', default='./runs', type=str)
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='keep a checkpoint for every epoch in addition to best.ckpt and last.ckpt')

    args = parser.parse_args()

    args.regression = "mse" in args.losses
    if args.regression and ({"ce", "mv"} & set(args.losses)):
        parser.error("mse (regression) cannot be combined with ce or mv (classification)")

    if args.batch_size < 2 and set(distance_losses) & set(args.losses):
        parser.error("Batch size should be at least 2 if you are using order, euc or order_p loss!")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split_dirs = {split: os.path.join(args.dataset, "HC", split) for split in ["train", "val", "test"]}
    train_ages = list_age_folders(split_dirs["train"])
    if not train_ages:
        parser.error(f"No age folders found in {split_dirs['train']}")
    if args.min_age is None:
        args.min_age = train_ages[0]
    if args.max_age is None:
        args.max_age = train_ages[-1]
    args.num_classes = 1 if args.regression else args.max_age - args.min_age + 1
    args.max_label_diff = args.max_age - args.min_age if args.order_weights == "age" else None

    print(args)

    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    training_folder = os.path.join(args.output_dir, f"{dataset_name}_{args.model_name}_{timestamp}")
    weights_folder = os.path.join(training_folder, "weights")
    os.makedirs(weights_folder)

    with open(os.path.join(training_folder, 'params.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    logger = Logger(os.path.join(training_folder, 'logs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args.model_name, args.num_classes)

    print("Training folder: ", training_folder)
    print("Model parameters: ", get_n_params(model))

    # Load datasets
    loader_kwargs = dict(min_age=args.min_age, max_age=args.max_age, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    dataset_train = get_loader(split_dirs["train"], mode="train", augment=not args.no_augmentation, **loader_kwargs)
    dataset_val = get_loader(split_dirs["val"], mode="val", **loader_kwargs)
    dataset_test = get_loader(split_dirs["test"], mode="test", **loader_kwargs)

    # Optimizer selection based on opt argument
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
    opt = optimizers[args.opt](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.to(device)

    best_path = os.path.join(weights_folder, "best.ckpt")
    best_mae, best_epoch = float("inf"), 0

    # Training loop
    for e in range(args.epochs):

        train(model, dataset_train, args, opt, e, device, logger)

        torch.save(model.state_dict(), os.path.join(weights_folder, "last.ckpt"))
        if args.save_every_epoch:
            torch.save(model.state_dict(), os.path.join(weights_folder, f"{e+1}.ckpt"))

        val_metrics, _ = evaluate(model, dataset_val, args, device)
        for tag, value in val_metrics.items():
            logger.scalar_summary("val/" + tag, value, e + 1)
        print(f"Testing on VAL Dataset at Epoch {e+1}: {format_metrics(val_metrics)}")

        if val_metrics["mae"] < best_mae:
            best_mae, best_epoch = val_metrics["mae"], e + 1
            torch.save(model.state_dict(), best_path)
            print(f"New best val MAE {best_mae:.4f}, saved {best_path}")
        elif args.patience and e + 1 - best_epoch >= args.patience:
            print(f"Early stopping: no val MAE improvement for {args.patience} epochs")
            break

    # Test the checkpoint with the best validation MAE
    if not os.path.exists(best_path):
        print("Val MAE never improved (NaN?); testing the last checkpoint instead")
        best_path = os.path.join(weights_folder, "last.ckpt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, test_predictions = evaluate(model, dataset_test, args, device)
    test_metrics["best_epoch"] = best_epoch
    for tag, value in test_metrics.items():
        logger.scalar_summary("test/" + tag, value, best_epoch)
    print(f"Testing on TEST Dataset with best epoch {best_epoch}: {format_metrics(test_metrics)}")

    with open(os.path.join(training_folder, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    test_predictions.to_csv(os.path.join(training_folder, 'test_predictions.csv'), index=False)
    logger.close()


if __name__ == '__main__':
    main()
