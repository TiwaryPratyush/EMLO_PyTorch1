import argparse
import lightning as L
from models.dog_classifier import DogClassifier
from datamodules.catdog import DogImageDataModule
from rich.console import Console
from pathlib import Path

console = Console()


def evaluate_model(checkpoint_path, batch_size=32, num_classes=120, num_workers=4):
    console.print("[bold green]Loading model from checkpoint...[/bold green]")
    
    # Ensure the checkpoint path exists
    if not Path(checkpoint_path).is_file():
        console.print(f"[bold red]Error: Checkpoint path {checkpoint_path} does not exist![/bold red]")
        return
    
    # Load the model from the checkpoint
    model = DogClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
    model.eval()

    # Set up the data module with correct `num_workers`
    data_module = DogImageDataModule(batch_size=batch_size, num_workers=num_workers)
    data_module.setup("test")  # Assuming we are using the validation set for evaluation

    # Create a trainer for evaluation
    trainer = L.Trainer(accelerator="auto")
    
    # Run the evaluation on the validation set
    console.print("[bold green]Running validation...[/bold green]")
    validation_results = trainer.validate(model, dataloaders=data_module.val_dataloader())
    
    # Print the validation metrics
    console.print("[bold yellow]Validation metrics:[/bold yellow]")
    for key, value in validation_results[0].items():
        console.print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on validation dataset")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    args = parser.parse_args()

    evaluate_model(args.ckpt_path, args.batch_size, num_workers=args.num_workers)
