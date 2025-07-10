import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='macro')
    recall = recall_score(test_labels, test_preds, average='macro')
    f1 = f1_score(test_labels, test_preds, average='macro')
    qwk = cohen_kappa_score(test_labels, test_preds, weights='quadratic')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, QWK: {qwk:.4f}")