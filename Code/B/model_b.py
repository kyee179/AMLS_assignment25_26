import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ResNet18_28x28(nn.Module):
    """
    ResNet-18 adapted for 28x28 Grayscale images (1 channel).
    """

    def __init__(self, num_classes=1):
        super(ResNet18_28x28, self).__init__()
        self.resnet = resnet18(weights=None)

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.resnet.maxpool = nn.Identity()

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ModelB:
    def __init__(self, learning_rate=0.001, epochs=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet18_28x28(num_classes=1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train(self, train_loader, val_loader):
        print(f"Training ResNet-18 on {self.device} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            avg_train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train

            val_loss, val_acc = self._validate(val_loader)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

        return self.history

    def _validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return running_loss / len(loader), correct / total

    def evaluate(self, loader, dataset_name="Test"):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        print(f"--- {dataset_name} Metrics (ResNet) ---")
        print(f"Acc: {acc:.4f}, Prec: {p:.4f}, Rec: {r:.4f}, F1: {f1:.4f}")
