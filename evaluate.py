from data_loaders_4binover import get_loader
import argparse, json
import os
import torch

from train import build_model, evaluate, format_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained brain age model')
    parser.add_argument('--run_dir', required=True, type=str,
                        help='training folder created by train.py (contains params.json and weights/)')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='folder with per-age subfolders, e.g. data/HC/test or an external cohort')
    parser.add_argument('--checkpoint', default='best.ckpt', type=str,
                        help='checkpoint file name inside run_dir/weights')
    parser.add_argument('--output', default=None, type=str,
                        help='predictions CSV (default: run_dir/predictions_<data_dir name>.csv)')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, 'params.json')) as f:
        train_args = argparse.Namespace(**json.load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(train_args.model_name, train_args.num_classes)
    checkpoint = os.path.join(args.run_dir, 'weights', args.checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    # Ages outside the training range are kept: they are only used to compute BrainAGE
    loader = get_loader(args.data_dir, train_args.min_age, train_args.max_age, batch_size=args.batch_size,
                        mode='test', num_workers=args.num_workers, keep_out_of_range=True)
    metrics, predictions = evaluate(model, loader, train_args, device, compute_loss=False)
    print(format_metrics(metrics))

    output = args.output
    if output is None:
        name = os.path.basename(os.path.normpath(args.data_dir))
        output = os.path.join(args.run_dir, f'predictions_{name}.csv')
    predictions.to_csv(output, index=False)
    with open(os.path.splitext(output)[0] + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved predictions to", output)


if __name__ == '__main__':
    main()
