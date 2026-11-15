import argparse
import torch
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--content_dir', type=str, default='C:\\Computer Science\\AI ML Projects\\Neural Style Transfer with AdaIN\\content_data', 
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='C:\\Computer Science\\AI ML Projects\\Neural Style Transfer with AdaIN\\style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='C:\\Computer Science\\AI ML Projects\\Neural Style Transfer with AdaIN\\vgg_normalized.pth',
                        help='Location of pre-trained VGG model')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')
    parser.add_argument('--final_size', type=int, default=256, help='Final size of the image')
    parser.add_argument('--content_size', type=int, default=512, help='Size of the content image')
    parser.add_argument('--style_size', type=int, default=512, help='Size of the style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Whether to crop the image')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5, help='Style weight')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=2, help='Save Interval')
    # --- Previous arguments (replaced) ---
    # parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
    # parser.add_argument('--decoder_path', type=str, default=None, help='Path to decoder checkpoint')
    # parser.add_argument('--optimizer_path', type=str, default=None, help='Path to optimizer checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g. experiment/exp1/checkpoint_2.pth)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    content_transform = get_transforms(args.content_size, args.crop, args.final_size)  # Define your content transformations here
    style_transform = get_transforms(args.style_size, args.crop, args.final_size)    # Define your style transformations here

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    print('Number of batches in content dataset: ', len(content_loader))
    print('Number of batches in style dataset: ', len(style_loader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(                                    # use to adjust the learning rates during training 
        optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    # --- Previous resume logic (replaced) ---
    # if args.resume:
    #     decoder.load_state_dict(torch.load(args.decoder_path))
    #     optimizer.load_state_dict(torch.load(args.optimizer_path))

    start_epoch = 0

    if args.resume and Path(args.resume).is_file():
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        # Apply command-line learning rate if user specifies a new one
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
        print(f"Successfully resumed from epoch {start_epoch}. Starting training from epoch {start_epoch + 1}.")

    print('Training...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()  #put your model on eval when you are not training the model

    running_loss = None
    running_closs = None
    running_sloss = None 

    # --- Previous loop (replaced) ---
    # for epoch in range(args.epochs):
    for epoch in range(start_epoch, args.epochs):
        progress_bar = tqdm(zip(content_loader, style_loader), total=min(len(content_loader),len(style_loader)))

        running_loss = 0
        running_closs = 0
        running_sloss = 0

        # --- Previous loop (replaced) ---
        # for content_batch, style_batch in progress_bar:
        for i, (content_batch, style_batch) in enumerate(progress_bar):
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # print(content_batch.shape)
            # print(style_batch.shape)
            
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0

            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss: {loss.item():.4f}, Content Loss: {loss_c.item():.4f}, Style Loss: {loss_s.item():.4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

            # --- Added: emergency backup checkpoint every 300 steps ---
            if (i + 1) % 300 == 0:
                torch.save({
                    'epoch': epoch,
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_dir / 'checkpoint_latest.pth')

        scheduler.step() 
        running_loss /= len(content_loader)
        running_closs /= len(content_loader)
        running_sloss /= len(content_loader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss: {running_loss:.4f}, Style Loss: {running_sloss:.4f}, Content Loss: {running_closs:.4f}')

            # print(len(c_feats))
            # print(len(s_feats))

            # print(c_feats[0].shape)

        if (epoch+1) % args.save_interval == 0:
            # --- Previous saving (replaced) ---
            # torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            # torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            checkpoint = {
                'epoch': epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, save_dir / f'checkpoint_{epoch+1}.pth')
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)


if __name__ == "__main__":
    main()