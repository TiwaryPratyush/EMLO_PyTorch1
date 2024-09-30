import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    EarlyStopping,  # Import EarlyStopping
)
from datamodules.catdog import DogImageDataModule  # Ensure your DataModule is properly referenced
from models.dog_classifier import DogClassifier  # Update based on your model structure
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
from utils.rich_utils import print_config_tree, print_rich_progress, print_rich_panel
from pathlib import Path
import os

log = get_pylogger(__name__)

@task_wrapper
def train():
    # Create the logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Set up data module with the right `num_workers`
    data_module = DogImageDataModule(data_dir="data", batch_size=32, num_workers=4)  # Increase num_workers

    # Set up model
    model = DogClassifier(lr=1e-3)

    # Set up logger
    logger = TensorBoardLogger(save_dir="logs", name="dog_breed_classification")

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='model_checkpoint',
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_weights_only=False,
    )

    # Set up early stopping (to avoid overfitting)
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        patience=3,  # Stop training if no improvement after 3 epochs
        mode="min"
    )

    # Set up other callbacks
    rich_progress_bar = RichProgressBar()
    rich_model_summary = RichModelSummary(max_depth=2)

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=30,  # Increased to 30 epochs for better learning
        callbacks=[checkpoint_callback, early_stopping_callback, rich_progress_bar, rich_model_summary],  # Added early stopping
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # Print config for debugging purposes
    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}
    print_config_tree(config, resolve=True, save_to_file=True)

    # Start training the model
    print_rich_panel("Starting training", "Training")
    trainer.fit(model, datamodule=data_module)

    # Evaluate the model (optional)
    print_rich_panel("Starting testing", "Testing")
    trainer.test(model, datamodule=data_module)

    # Final output
    print_rich_progress("Finishing up")

if __name__ == "__main__":
    train()