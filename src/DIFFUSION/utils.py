import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 128
TIME_STEPS = 450


crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(IMAGE_SIZE)
])

crop_and_normalize = transforms.Compose([
    crop,
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])