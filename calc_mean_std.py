import torch
from PIL import Image
import torchvision.transforms as transforms
import glob

raw_images_path = glob.glob("dataset/**/*.jpg", recursive=True)


compose = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


num_images = len(raw_images_path)
mean = 0
std = 0
for path in raw_images_path:
    im = Image.open(path)
    im = compose(im)
    mean += torch.mean(torch.mean(im,1),1)
    std += torch.std(im)

mean /= num_images
std /= num_images

print(mean)
print(std)