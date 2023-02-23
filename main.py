from app.trainer.funiture_dataset import FurnitureDataset
from app.trainer.model import FurnitureClassifier
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--data_path', default="./dataset/furniture_dataset", help='path to the image folders directory')
    add_arg('--lr', default=1e-4, type=float, help="learning rate")
    add_arg('--batch_size', default=8, type=int, help="batch size for test, train, and validation")
    add_arg('--mode', default='test', choices=['train', 'test', 'serve'], help="decides which process to starts")
    add_arg('--model_path', default="./output/model", help='path to the trained model directory')
    add_arg('--save_path', default="./output/model", help='path to save the trained model')
    add_arg('--port', default='8080', type=str, help="default port for serving API")

    return parser.parse_args()


def train(train_loader, valid_loader, model):
    # Initialize the CNN model and define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model for 10 epochs
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        val_acc, val_loss = validation(valid_loader, model=model)
        print('[Epoch %d] Validation Loss: %.3f \t Validation Accuracy: %.3f' % (epoch + 1, val_loss, val_acc))


def validation(val_loader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / len(val_loader)

    return val_accuracy, val_loss


def test(test_loader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

        test_accuracy = test_correct / len(test_loader)

    print('Test Loss: %.3f \t Test Accuracy: %.3f' % (test_loss, test_accuracy))


def load_data(mode):
    dataset = FurnitureDataset(path=args.data_path)
    ds_len = len(dataset)
    train_dataset, validation_dataset, test_dataset = data.random_split(dataset,
                                                                        [ds_len * 0.8, ds_len * 0.1, ds_len * 0.1])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if mode == 'train':
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        return None, None, test_dataloader


def save_model(save_path, model):
    torch.save(model, save_path)


def load_model(model_path, num_classes):
    new_model = FurnitureClassifier(num_classes=num_classes)
    if model_path:
        new_model = torch.load(model_path)
    return new_model


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train_loader, val_loader, test_loader = load_data(args.mode)
        model = load_model(model_path=args.model_path, num_classes=train_loader.dataset.num_classes())
        train(train_loader=train_loader, valid_loader=val_loader, model=model)
        test(test_loader=test_loader, model=model)
        save_model(save_path=args.save_path, model=model)
    elif args.mode == 'test':
        _, _, test_loader = load_data(args.mode)
        model = load_model(model_path=args.model_path, num_classes=test_loader.dataset.num_classes())
        test(test_loader=test_loader, model=model)
    else:
        # serve AIP
        pass



