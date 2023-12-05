import torch
from torch.utils.data import DataLoader
from .preprocessing import Eye_Dataset
from metrics import specificity, accuracy, calc_iou
from .models.model import MyFrame
from .models.encoder.unet import UTNet

def validate(model, val_loader):
    model.eval()
    total_specificity = 0
    total_accuracy = 0
    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device)
            mask = mask.to(device)
            pred = model.forward(img)

            # Calculate metrics
            spec = specificity(pred, mask)
            acc = accuracy(pred, mask)
            iou = calc_iou(pred, mask)

            total_specificity += spec.item()
            total_accuracy += acc.item()
            total_iou += iou.item()
            total_samples += 1

    avg_specificity = total_specificity / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_iou = total_iou / total_samples

    print(f'Validation Specificity: {avg_specificity}')
    print(f'Validation Accuracy: {avg_accuracy}')
    print(f'Validation IoU: {avg_iou}')

if __name__ == "__main__":
    root_path = '/Datasets/CVC Clinicdb'
    batch_size = 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    val_dataset = Eye_Dataset(root_path, 'Validation')  # You might need to adjust this based on your data setup
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MyFrame(UTNet, learning_rate=0.0002, device=device)
    model.load('/Saved Models/CVC_UT_300.pth')  # Load the trained model

    validate(model, val_loader)
