from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch


class TranslationDataset(Dataset):
    # Paired translation hence data needs to be of same length
    def __init__(self, root_x: str, root_y: str, transform=None):
        self.root_x = Path(root_x)
        self.root_y = Path(root_y)
        self.transform = transform

        self.x_samples = sorted([sample for sample in self.root_x.iterdir() if sample.is_file()])
        self.y_samples = sorted([sample for sample in self.root_y.iterdir() if sample.is_file()])

        if len(self.x_samples) != len(self.y_samples):
            print('Warning: Paired translation is being performed as default, '
                  'hence if data is not of same length set LAMBDA-Identity to 0')
        self.x_len = len(self.x_samples)
        self.y_len = len(self.y_samples)
        self.data_len = max(self.x_len, self.y_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        x_path, y_path = self.x_samples[idx % self.x_len], self.y_samples[idx % self.y_len]
        x_img, y_img = Image.open(x_path).convert('RGB'), Image.open(y_path).convert('RGB')

        if self.transform:
            return self.transform(x_img), self.transform(y_img)
        else:
            return x_img, y_img


def save_model(network, optimizer, filename='model_checkpoint.pth.tar'):
    print(f'-Saving Network at {filename}')
    checkpoint = {
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_model(filename, network, optimizer=None, lr=None, device='cpu'):
    print(f'-Loading Network from {filename}')
    checkpoint = torch.load(filename, map_location=device)
    network.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in optimizer.param_group:
            param_group['lr'] = lr
