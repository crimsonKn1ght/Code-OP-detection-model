from torchvision import transforms, datasets


def get_image_dataset(image_folder, image_size=(128, 128)):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)

    images, labels = [], []
    for img_pil, label in dataset:
        images.append(img_pil.squeeze().numpy())
        labels.append(label)

    return images, labels