import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from lenet import LeNet5

if __name__ == "__main__":
    mnist_transforms = transforms.Compose(
        # 0.1307 和 0.3081 分别是 MNIST 数据集的像素值的均值和标准差
        [transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]
    )
    train_val_dataset = datasets.MNIST(
        root="./datasets/", train=True, download=True, transform=mnist_transforms
    )
    test_dataset = datasets.MNIST(
        root="./datasets/", train=False, download=True, transform=mnist_transforms
    )

    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=train_val_dataset, lengths=[train_size, val_size]
    )

    BATCH_SIZE = 32

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    le_net5 = LeNet5()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=le_net5.parameters(), lr=0.001)
    accuracy = Accuracy(task="multiclass", num_classes=10)

    EPOCHS = 12
    last_val_acc = 0.0

    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_dataloader:
            le_net5.train()

            y_pred = le_net5(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            acc = accuracy(y_pred, y)
            train_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        val_loss, val_acc = 0.0, 0.0
        le_net5.eval()
        with torch.inference_mode():
            for X, y in val_dataloader:
                y_pred = le_net5(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()

                acc = accuracy(y_pred, y)
                val_acc += acc

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        if val_acc > last_val_acc:
            torch.save(le_net5.state_dict(), "model.pth")
        last_val_acc = val_acc
