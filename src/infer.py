import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from models.dog_classifier import DogClassifier
from datamodules.catdog import DogImageDataModule
import random
from rich.console import Console

console = Console()


def inference(model, img_tensor, class_labels):
    # Move the input tensor to the same device as the model
    img_tensor = img_tensor.unsqueeze(0).to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Get the predicted class label and confidence
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return predicted_label, confidence


def visualize_prediction(img, predicted_label, confidence):
    # Convert the tensor to a NumPy array for visualization
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # Display the image with the predicted label and confidence
    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label.capitalize()} (Confidence: {confidence:.2f})")
    plt.show()


def main(checkpoint_path, batch_size=32, num_classes=120,num_workers = 4):
    console.print("[bold green]Loading model from checkpoint...[/bold green]")

    # Load the model from the checkpoint
    model = DogClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the data module
    data_module = DogImageDataModule(batch_size=batch_size,num_workers=num_workers)
    data_module.setup("fit")  # Assuming we want to use the validation set

    # Get the validation dataloader
    val_dataloader = data_module.val_dataloader()

    # Define class labels (replace with actual dog breed names if available)
    class_labels = [f"breed_{i}" for i in range(num_classes)]

    # Collect 10 random samples from the validation dataset
    samples = random.sample(list(val_dataloader.dataset), 10)

    # Run inference on the 10 random samples
    console.print("[bold green]Running inference on 10 images...[/bold green]")
    for img, label in samples:
        predicted_label, confidence = inference(model, img, class_labels)
        console.print(f"Predicted: {predicted_label}, Confidence: {confidence:.2f}")
        visualize_prediction(img, predicted_label, confidence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on 10 random images from the validation dataset")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for validation"
    )
    args = parser.parse_args()

    main(args.ckpt_path, args.batch_size)