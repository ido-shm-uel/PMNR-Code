#!/usr/bin/env python3

import argparse
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import v2

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
SEED = 1

MNIST_INPUT_SQRT = 28
MNIST_INPUT = MNIST_INPUT_SQRT * MNIST_INPUT_SQRT
MNIST_OUTPUT = 10

LEAKY_RELU_SLOPE = 0.1

LEAKY_RELU_LAYERS = 14
LEAKY_RELU_LAYER = 28
BILINEAR_LAYER_SQRT = 8
BILINEAR_LAYER = BILINEAR_LAYER_SQRT * BILINEAR_LAYER_SQRT
RELU_LAYERS = [256, 144, 64, 36]
PERCENTAGE = 0.01

class BilinearLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class LeakyRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = LEAKY_RELU_SLOPE
        self.layers = LEAKY_RELU_LAYERS
        self.leaky_relu_layer = LEAKY_RELU_LAYER
        self.fc1 = torch.nn.Linear(MNIST_INPUT, self.leaky_relu_layer)
        self.fc2 = torch.nn.Linear(self.leaky_relu_layer, self.leaky_relu_layer)
        self.fc3 = torch.nn.Linear(self.leaky_relu_layer, MNIST_OUTPUT)

    def forward(self, x):
        x = self.fc1(x.reshape(-1, MNIST_INPUT))
        for _ in range(self.layers-1):
            x = self.fc2(torch.nn.functional.leaky_relu(x, self.slope))
        return self.fc3(torch.nn.functional.leaky_relu(x, self.slope))


class LeakyReluSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = LEAKY_RELU_SLOPE
        self.leaky_relu_layer = LEAKY_RELU_LAYER
        self.fc1 = torch.nn.Linear(MNIST_INPUT, self.leaky_relu_layer)
        self.fc2 = torch.nn.Linear(self.leaky_relu_layer, self.leaky_relu_layer)
        self.fc3 = torch.nn.Linear(self.leaky_relu_layer, MNIST_OUTPUT)

    def forward(self, x):
        x = self.fc1(x.reshape(-1, MNIST_INPUT))

        x = self.fc2(torch.nn.functional.leaky_relu(x, self.slope))
        x = torch.nn.functional.leaky_relu(x, self.slope)
        x = self.fc2(torch.nn.functional.sigmoid(x))

        x = self.fc2(torch.nn.functional.relu(x))
        x = torch.nn.functional.leaky_relu(x, self.slope)
        x = self.fc2(torch.nn.functional.sigmoid(x))

        x = self.fc2(torch.nn.functional.leaky_relu(x, self.slope))
        x = torch.nn.functional.leaky_relu(x, self.slope)
        x = self.fc2(torch.nn.functional.sigmoid(x))

        return self.fc3(torch.nn.functional.leaky_relu(x, self.slope))


class ReluBilinearSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bilinear_layer_sqrt = BILINEAR_LAYER_SQRT
        self.bilinear_layer = BILINEAR_LAYER
        self.fc1 = torch.nn.Linear(MNIST_INPUT, self.bilinear_layer)
        self.fc2 = torch.nn.Linear(self.bilinear_layer, MNIST_OUTPUT)
        self.bilinear = BilinearLayer()

    def forward(self, x):
        x = self.fc1(x.reshape(-1, MNIST_INPUT))
        x = (torch.nn.functional.relu(x)).reshape(-1,
                                                  self.bilinear_layer_sqrt,
                                                  self.bilinear_layer_sqrt)
        x = self.fc2((self.bilinear(x, x)).reshape(-1, self.bilinear_layer))
        return torch.nn.functional.softmax(x, dim=1)


class Relu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_layers = [256, 144, 64, 36]
        self.fc1 = torch.nn.Linear(MNIST_INPUT, self.relu_layers[0])
        self.fc2 = torch.nn.Linear(self.relu_layers[0], self.relu_layers[1])
        self.fc3 = torch.nn.Linear(self.relu_layers[1], self.relu_layers[2])
        self.fc4 = torch.nn.Linear(self.relu_layers[2], self.relu_layers[-1])
        self.fc5 = torch.nn.Linear(self.relu_layers[-1], MNIST_OUTPUT)

    def forward(self, x):
        x = self.fc1(x.reshape(-1, MNIST_INPUT))
        x = self.fc2(torch.nn.functional.relu(x))
        x = self.fc3(torch.nn.functional.relu(x))
        x = self.fc4(torch.nn.functional.relu(x))
        return self.fc5(torch.nn.functional.relu(x))


def train(model, device, train_loader, optimizer):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    tqdm.write(f"Train Epoch: {epoch}\tAverage test loss: {test_loss:.4f}, Accuracy: "
        +f"{correct}/{len(test_loader.dataset)} "
        +f"({correct / (PERCENTAGE * len(test_loader.dataset)):.0f}%).")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="MNIST Classifier Training")
    parser.add_argument(
        "--epochs",
        type=int,
        default=33,
        metavar="N",
        help="number of epochs to train (default: 33)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=True, help="disables CUDA training"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="Relu",
        metavar="N",
        help="Network architecture (relu|leaky_relu|"+
        "relu_bilinear_softmax|leaky_relu_sigmoid). Default: relu.",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="",
        metavar="N",
        help="Output directory.",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "datasets",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), v2.ToDtype(torch.float32, scale=True)]
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "datasets",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), v2.ToDtype(torch.float32, scale=True)]
            ),
        ),
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        **kwargs
    )

    if args.architecture == "leaky_relu":
        model = LeakyRelu().to(device)
    elif args.architecture == "leaky_relu_sigmoid":
        model = LeakyReluSigmoid().to(device)
    elif args.architecture == "relu_bilinear_softmax":
        model = ReluBilinearSoftmax().to(device)
    else:
        model = Relu().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Train model.
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader, epoch)

    # Export model.
    print("Saving Model...")
    input_names = ["Image"]
    output_names = ["Classification"]
    dummy_input = torch.randn(1, 1, MNIST_INPUT_SQRT, MNIST_INPUT_SQRT)
    torch.onnx.export(
        model,
        dummy_input,
        f"{args.directory}/mnist_{args.architecture}.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    torch.save(model.state_dict(), f"{args.directory}/mnist_{args.architecture}_cnn.pt")


if __name__ == "__main__":
    main()
