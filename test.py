from generator import Generator
from dataloading import load_model
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(256),
    T.Normalize(mean=.5, std=.5)
])


gen = Generator()
load_model('generator_checkpoint-9.pth.tar', gen)

with torch.no_grad():
    gen.eval()
    path = 'Data/Test-Sketches/f1-005-01-sz1.jpg'
    inputs = Image.open(path).convert('RGB')
    inputs = transform(inputs)
    output = gen(inputs.unsqueeze(0))

plt.imshow(inputs.squeeze().permute(1, 2, 0))
plt.show()
plt.imshow(output.squeeze().permute(1, 2, 0))
plt.show()