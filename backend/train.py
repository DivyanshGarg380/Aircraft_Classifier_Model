import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

from app.model import build_model, freeze_backbone, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_loaders(data_dir: str, batch_size: int):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=eval_transform)

    assert train_ds.classes == CLASSES, (
        f"Class mismatch! ImageFolder found {train_ds.classes}, "
        f"expected {CLASSES}. Check your data/ folder names."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze", action="store_true", help="freeze backbone, only train the final layer (faster, good for small datasets)")
    args = parser.parse_args()

    train_loader, val_loader = get_loaders(args.data_dir, args.batch_size)

    model = build_model(pretrained=True).to(DEVICE)
    if args.freeze:
        freeze_backbone(model)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    best_val_acc = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/aircraft_classifier.pt")
            print(f"  -> new best model saved (val_acc={val_acc:.3f})")

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.3f}")

    test_dir = Path(args.data_dir) / "test"
    if test_dir.exists():
        test_ds = datasets.ImageFolder(str(test_dir), transform=eval_transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        model.load_state_dict(torch.load("models/aircraft_classifier.pt"))
        _, test_acc = run_epoch(model, test_loader, criterion)
        print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()