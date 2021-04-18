import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import ViT
from utils import load_data, WarmupCosineSchedule, get_config
import os
import argparse
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', required=True, type=str,
        help='Path to training dataset')
    parser.add_argument('--val_dir', required=True, type=str,
        help='Path to validation dataset')

    parser.add_argument('--batch_size', required=False, type=int, default=1,
        help='The batch size of training data')
    parser.add_argument('--val_batch_size', required=False, type=int, default=1,
        help='The batch size of validation data')
    parser.add_argument('--image_size', required=False, type=int, default=224,
        help='Size of images')

    parser.add_argument('--ckpth', required=False, type=str, default='./',
        help='Path for storing the checkpoints')
    parser.add_argument('--continue_train', required=False, type=bool,
        default=False, help='Continue the training')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
        default="ViT-B_16",
        help="Which variant to use.")

    parser.add_argument("--num_steps", default=10000, type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument('--n_classes', required=False, type=int, default=1000,
        help='Number of classes in the data')
    parser.add_argument('--lr', required=False, type=float, default=0.003,
        help='The learning rate for the optimizer')
    parser.add_argument("--weight_decay", required=False, type=float, default=0.1,
        help='The decay to apply to weights in training')
    parser.add_argument("--warmup_steps", default=1000, type=int,
        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    return parser.parse_args()



def train(model, args):
    if args.continue_train:
        try:
            model.load_state_dict(torch.load(os.path.join(args.ckpth, f'{args.model_type}.pth')))
            print('Continuing training')
        except:
            print('Starting from scratch')
    else:
        print('Starting from scratch')

    loss_obj = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(opt, args.warmup_steps, args.num_steps)
   
    train_loader, val_loader = load_data(args.train_dir, args.val_dir, image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size, val_batch_size=args.val_batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.zero_grad()
    step = 0
    while True:
        model.train()

        tqdm_train = tqdm(train_loader,
                        desc="Training (X / X Steps) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
        tqdm_val = tqdm(val_loader, 
                        desc="Validating (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)

        for batch in tqdm_train:
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            logits = model(x)
            loss = loss_obj(logits.view(-1, args.n_classes), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            tqdm_train.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (step, args.num_steps, loss.item()))
            step += 1
            if step == args.num_steps:
                break
        
        model.train(mode=False)
        for batch in tqdm_val:
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            logits = model(x)
            val_loss = loss_obj(logits.view(-1, args.n_classes), y.view(-1))
            tqdm_val.set_description("Validating (loss=%2.5f)" % (val_loss.item()))

        torch.save(model.state_dict(), os.path.join(args.ckpth, f'{args.model_type}.pth'))
        if step == args.num_steps:
                break


def main():
    args = get_arguments()
    config = get_config(args.model_type)
    model = ViT(image_size=args.image_size, patch_size=config.patches, n_classes=args.n_classes, 
            depth=config.transformer.depth, dim=config.dim, heads=config.transformer.heads, 
            attention_dropout=config.transformer.attention_dropout, dropout=config.transformer.dropout,
            mlp_hidden=config.transformer.mlp_hidden)
    train(model, args)


if __name__ == "__main__":
    main()