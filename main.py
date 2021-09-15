from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from dataloading import TranslationDataset, save_model, load_model
import torchvision.transforms as T
import torch.optim as optim, torch.nn as nn
from tqdm.auto import tqdm
import torch
from torchvision.utils import save_image

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),   
    T.Normalize(mean=.5, std=.5)
])
dataset = TranslationDataset('Data/Sketches', 'Data/Photos', transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


def train_fn(gen: Generator, disc: Discriminator, data_loader: DataLoader, opt_d, opt_g, criterion, epoch):
    lambda_identity = 2
    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

    for batch_idx, (sketch, photo) in loop:
        # Training the discriminator
        # On the real samples should be close to all ones
        real_preds = disc(photo)  # 3x3 = 3x3 1s
        disc_real = criterion(real_preds, torch.ones_like(real_preds))

        # On the fake samples
        fake = gen(sketch)  # 3, 256, 256
        fake_preds = disc(fake.detach())
        disc_fake = criterion(fake_preds, torch.zeros_like(fake_preds))

        # Adversarial loss - Discriminator
        disc_loss = (disc_real + disc_fake) / 2
        opt_d.zero_grad()
        disc_loss.backward()
        opt_d.step()

        # Training the generator
        disc_fake = disc(fake)
        gen_adv_loss = criterion(disc_fake, torch.ones_like(disc_fake))  # Generator adversarial loss

        # Identity loss
        gen_id_loss = criterion(photo, fake)
        gen_loss = gen_adv_loss + lambda_identity * gen_id_loss

        opt_g.zero_grad()
        gen_loss.backward()
        opt_g.step()

        loop.set_description(f'Step [{batch_idx+1}/{len(data_loader)}]')
        loop.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

        if batch_idx % 5 == 0:
            save_image(fake.detach().squeeze(), f'Samples/Fake-{epoch+1}-{batch_idx}.png')


if __name__ == '__main__':
    n_epochs = 10
    generator = Generator()
    discriminator = Discriminator()
    opt_gen = optim.Adam(generator.parameters(), betas=(0.5, 0.999), lr=1e-5)
    opt_disc = optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=1e-5)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        train_fn(generator, discriminator, loader, opt_disc, opt_gen, loss_fn, epoch)
        save_model(generator, opt_gen, f'generator_checkpoint-{epoch+1}.pth.tar')
        save_model(discriminator, opt_disc, f'discriminator_checkpoint-{epoch + 1}.pth.tar')
