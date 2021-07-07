import PIL
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Replicating the results of COIN.")
parser.add_argument("--image_dir", help="path to image you want to compress")
parser.add_argument(
    "--iters",
    default=50000,
    type=int,
    help="total number of iterations (minibatches) to fit for",
)
parser.add_argument("--batch_size", default=1, type=int)
args = parser.parse_args()


##################################################### DATALOADER
class COINDataset(Dataset):
    def __init__(self, img_dir):
        transform = ToTensor()
        self.image = transform(PIL.Image.open(img_dir))
        self.num_rows = self.image.shape[1]
        self.num_cols = self.image.shape[2]

    def __len__(self):
        return self.image[0].numel()

    def __getitem__(self, idx):

        row_idx = idx // self.num_cols
        col_idx = idx % self.num_cols

        return (row_idx, col_idx), self.image[:, row_idx, col_idx]


##################################################### MODEL
class MLP(nn.Module):
    def __init__(self, hidden_units=100):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(2, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, hidden_units)
        self.layer4 = nn.Linear(hidden_units, hidden_units)
        self.layer5 = nn.Linear(hidden_units, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.layer5(out)


if __name__ == "__main__":

    data = COINDataset(args.image_dir)
    dataloader = DataLoader(data, batch_size=args.batch_size)
    model = MLP()

    distortion_measure = nn.MSELoss()
    optimiser = optim.Adam(model.parameters())

    iterloader = iter(dataloader)
    for iter in range(args.iters):
        try:
            inputs, labels = next(iterloader)
        except StopIteration:
            iterloader = iter(dataloader)
            inputs, labels = next(iterloader)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = distortion_measure(outputs, labels)
        loss.backward()
        optimiser.step()

        # print statistics
        running_loss += loss.item()
        if iter % 2000 == 1999:  # print every 2000 mini-batches
            print(f"iter: {iter + 1}, loss: {running_loss / 2000}")
            running_loss = 0.0

    # save the weights
    torch.save(model.state_dict(), args.image_dir + ".t7")

    print("Reconstructing the image...")

    def reconstruct_image(model, dataloader, data):

        channels = 3
        num_rows = data.num_rows
        num_cols = data.num_cols

        reconstructed_image = torch.zeros((channels, num_rows, num_cols))

        for pixel_loc, rgb in tqdm(dataloader):
            reconstructed_image[:, pixel_loc[0], pixel_loc[1]] = model(pixel_loc)

        return reconstructed_image

    reconstructed_image = reconstruct_image(model, dataloader, data)

    # measure PSNR
