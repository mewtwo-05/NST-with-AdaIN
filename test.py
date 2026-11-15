import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

Image.MAX_IMAGE_PIXELS = None


def test_transform(size=512, crop=False):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def style_transfer(vgg, decoder, content_tensor, style_tensor, alpha=1.0):
    with torch.no_grad():
        c_feat = vgg(content_tensor, is_test=True)
        s_feat = vgg(style_tensor, is_test=True)

        t = adaptive_instance_normalization(c_feat, s_feat)
        t = alpha * t + (1.0 - alpha) * c_feat

        output = decoder(t)
        return output.clamp(0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Test your trained AdaIN style transfer model.")
    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--decoder', type=str, default='experiment/experiment_full/decoder_10.pth',
                        help='Path to trained decoder weights')
    parser.add_argument('--vgg', type=str, default='vgg_normalized.pth',
                        help='Path to VGG encoder weights')
    parser.add_argument('--output', type=str, default='stylized_result.png',
                        help='Path to save output image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Resize content image to this size (0 to keep original size)')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Resize style image to this size (0 to keep original size)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Style strength between 0.0 (content only) and 1.0 (full style)')
    parser.add_argument('--comparison', action='store_true',
                        help='Save a comparison grid [Content | Style | Output]')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vgg = VGGEncoder(args.vgg).to(device)
    vgg.eval()

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))
    decoder.eval()

    content_img = Image.open(args.content).convert('RGB')
    style_img = Image.open(args.style).convert('RGB')

    c_tf = test_transform(args.content_size)
    s_tf = test_transform(args.style_size)

    content_tensor = c_tf(content_img).unsqueeze(0).to(device)
    style_tensor = s_tf(style_img).unsqueeze(0).to(device)

    print(f"Stylizing with style strength (alpha) = {args.alpha}...")
    result = style_transfer(vgg, decoder, content_tensor, style_tensor, alpha=args.alpha)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.comparison:
        comp_tf = test_transform(512, crop=True)
        c_comp = comp_tf(content_img).unsqueeze(0).to(device)
        s_comp = comp_tf(style_img).unsqueeze(0).to(device)
        res_comp = style_transfer(vgg, decoder, c_comp, s_comp, alpha=args.alpha)
        grid = torch.cat([c_comp, s_comp, res_comp], dim=0)
        save_image(grid, output_path, nrow=3)
        print(f"Saved side-by-side comparison to: {output_path}")
    else:
        save_image(result, output_path)
        print(f"Saved stylized image to: {output_path}")


if __name__ == "__main__":
    main()
