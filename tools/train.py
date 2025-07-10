import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_qwk = cohen_kappa_score(train_labels, train_preds, weights='quadratic')

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validating"):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_qwk = cohen_kappa_score(val_labels, val_preds, weights='quadratic')

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1} - Current LR: {param_group['lr']:.6f}")

        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, QWK: {train_qwk:.4f}")
        print(f"  Val   Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, QWK: {val_qwk:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Validation Accuracy: {best_acc:.4f}")

