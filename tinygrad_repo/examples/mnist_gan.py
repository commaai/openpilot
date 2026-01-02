from pathlib import Path
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.helpers import trange
from tinygrad.nn import optim
from extra.datasets import fetch_mnist

class LinearGen:
  def __init__(self):
    self.l1 = Tensor.scaled_uniform(128, 256)
    self.l2 = Tensor.scaled_uniform(256, 512)
    self.l3 = Tensor.scaled_uniform(512, 1024)
    self.l4 = Tensor.scaled_uniform(1024, 784)

  def forward(self, x):
    x = x.dot(self.l1).leaky_relu(0.2)
    x = x.dot(self.l2).leaky_relu(0.2)
    x = x.dot(self.l3).leaky_relu(0.2)
    x = x.dot(self.l4).tanh()
    return x

class LinearDisc:
  def __init__(self):
    self.l1 = Tensor.scaled_uniform(784, 1024)
    self.l2 = Tensor.scaled_uniform(1024, 512)
    self.l3 = Tensor.scaled_uniform(512, 256)
    self.l4 = Tensor.scaled_uniform(256, 2)

  def forward(self, x):
    # balance the discriminator inputs with const bias (.add(1))
    x = x.dot(self.l1).add(1).leaky_relu(0.2).dropout(0.3)
    x = x.dot(self.l2).leaky_relu(0.2).dropout(0.3)
    x = x.dot(self.l3).leaky_relu(0.2).dropout(0.3)
    x = x.dot(self.l4).log_softmax()
    return x

def make_batch(images):
  sample = np.random.randint(0, len(images), size=(batch_size))
  image_b = images[sample].reshape(-1, 28*28).astype(np.float32) / 127.5 - 1.0
  return Tensor(image_b)

def make_labels(bs, col, val=-2.0):
  y = np.zeros((bs, 2), np.float32)
  y[range(bs), [col] * bs] = val  # Can we do label smoothing? i.e -2.0 changed to -1.98789.
  return Tensor(y)

def train_discriminator(optimizer, data_real, data_fake):
  real_labels = make_labels(batch_size, 1)
  fake_labels = make_labels(batch_size, 0)
  optimizer.zero_grad()
  output_real = discriminator.forward(data_real)
  output_fake = discriminator.forward(data_fake)
  loss_real = (output_real * real_labels).mean()
  loss_fake = (output_fake * fake_labels).mean()
  loss_real.backward()
  loss_fake.backward()
  optimizer.step()
  return (loss_real + loss_fake).numpy()

def train_generator(optimizer, data_fake):
  real_labels = make_labels(batch_size, 1)
  optimizer.zero_grad()
  output = discriminator.forward(data_fake)
  loss = (output * real_labels).mean()
  loss.backward()
  optimizer.step()
  return loss.numpy()

if __name__ == "__main__":
  # data for training and validation
  images_real = np.vstack(fetch_mnist()[::2])
  ds_noise = Tensor.randn(64, 128, requires_grad=False)
  # parameters
  epochs, batch_size, k = 300, 512, 1
  sample_interval = epochs // 10
  n_steps = len(images_real) // batch_size
  # models and optimizer
  generator = LinearGen()
  discriminator = LinearDisc()
  # path to store results
  output_dir = Path(".").resolve() / "outputs"
  output_dir.mkdir(exist_ok=True)
  # optimizers
  optim_g = optim.Adam(get_parameters(generator),lr=0.0002, b1=0.5)  # 0.0002 for equilibrium!
  optim_d = optim.Adam(get_parameters(discriminator),lr=0.0002, b1=0.5)
  # training loop
  Tensor.training = True
  for epoch in (t := trange(epochs)):
    loss_g, loss_d = 0.0, 0.0
    for _ in range(n_steps):
      data_real = make_batch(images_real)
      for step in range(k):  # Try with k = 5 or 7.
        noise = Tensor.randn(batch_size, 128)
        data_fake = generator.forward(noise).detach()
        loss_d += train_discriminator(optim_d, data_real, data_fake)
      noise = Tensor.randn(batch_size, 128)
      data_fake = generator.forward(noise)
      loss_g += train_generator(optim_g, data_fake)
    if (epoch + 1) % sample_interval == 0:
      fake_images = generator.forward(ds_noise).detach().numpy()
      fake_images = (fake_images.reshape(-1, 1, 28, 28) + 1) / 2  # 0 - 1 range.
      save_image(make_grid(torch.tensor(fake_images)), output_dir / f"image_{epoch+1}.jpg")
    t.set_description(f"Generator loss: {loss_g/n_steps}, Discriminator loss: {loss_d/n_steps}")
  print("Training Completed!")
