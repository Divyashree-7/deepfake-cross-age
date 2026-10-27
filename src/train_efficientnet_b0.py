
from pathlib import Path
import copy
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(r"C:\Users\Divyashree\Downloads\Dataset")
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_DIR = DATA_DIR / "Train"
VAL_DIR = DATA_DIR / "Validation"
TEST_DIR = DATA_DIR / "Test"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "efficientnet_b0_deepfake_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 0.0001
SEED = 42

# First, test that the complete pipeline works.
SMOKE_TEST = True

# Smoke-test sample counts PER CLASS.
SMOKE_TRAIN_PER_CLASS = 1000
SMOKE_VAL_PER_CLASS = 250
SMOKE_TEST_PER_CLASS = 250


# Used for full training when SMOKE_TEST = False.
EPOCHS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["Fake", "Real"]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. CHECK DATASET FOLDERS
# ============================================================

def check_dataset_folders():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for class_name in CLASS_NAMES:
            class_dir = folder / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

    print("\nAll dataset folders found.")


# ============================================================
# 5. CREATE BALANCED SUBSETS FOR SMOKE TEST
# ============================================================

def make_balanced_subset(dataset, samples_per_class):
    """
    Select an equal number of Fake and Real images.
    Uses a fixed seed so the selection is reproducible.
    """

    rng = random.Random(SEED)
    selected_indices = []

    for class_index, class_name in enumerate(dataset.classes):

        class_indices = [
            index
            for index, label in enumerate(dataset.targets)
            if label == class_index
        ]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Not enough {class_name} images. "
                f"Found {len(class_indices)}."
            )

        rng.shuffle(class_indices)

        selected_indices.extend(
            class_indices[:samples_per_class]
        )

    rng.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


# ============================================================
# 6. LOAD DATASETS AND DATALOADERS
# ============================================================

def load_datasets():
    check_dataset_folders()

    # Keep references to the original datasets.
    original_train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform
    )

    original_val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=eval_transform
    )

    original_test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=eval_transform
    )

    expected_classes = {"Fake", "Real"}

    if set(original_train_dataset.classes) != expected_classes:
        raise ValueError(
            f"Unexpected classes: {original_train_dataset.classes}"
        )

    class_to_idx = original_train_dataset.class_to_idx

    if (
        class_to_idx != original_val_dataset.class_to_idx
        or class_to_idx != original_test_dataset.class_to_idx
    ):
        raise ValueError(
            "Class mappings differ between dataset splits."
        )

    print("\nClass mapping:", class_to_idx)

    print(
        "Full training images:",
        len(original_train_dataset)
    )
    print(
        "Full validation images:",
        len(original_val_dataset)
    )
    print(
        "Full test images:",
        len(original_test_dataset)
    )

    if SMOKE_TEST:
        train_dataset = make_balanced_subset(
            original_train_dataset,
            SMOKE_TRAIN_PER_CLASS
        )

        val_dataset = make_balanced_subset(
            original_val_dataset,
            SMOKE_VAL_PER_CLASS
        )

        test_dataset = make_balanced_subset(
            original_test_dataset,
            SMOKE_TEST_PER_CLASS
        )

        print("\nSMOKE TEST MODE ENABLED")
        print("Training samples:", len(train_dataset))
        print("Validation samples:", len(val_dataset))
        print("Test samples:", len(test_dataset))

    else:
        train_dataset = original_train_dataset
        val_dataset = original_val_dataset
        test_dataset = original_test_dataset

        print("\nFULL DATASET MODE ENABLED")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # Return the ORIGINAL dataset's class mapping.
    # Subset objects do not expose class_to_idx.
    return (
        train_loader,
        val_loader,
        test_loader,
        class_to_idx
    )


# ============================================================
# 7. CREATE EFFICIENTNET-B0 MODEL
# ============================================================

def create_model():
    print("\nLoading pretrained EfficientNet-B0...")

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    # Freeze the pretrained feature extractor for the first stage.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the original ImageNet classifier with two classes.
    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    model = model.to(DEVICE)

    print("Architecture: EfficientNet-B0")
    print("Device:", DEVICE)
    print("Output classes:", CLASS_NAMES)
    print("Feature extractor: frozen")

    return model


# ============================================================
# 8. TRAIN OR EVALUATE ONE EPOCH
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()

        # Keep frozen feature layers in evaluation mode.
        model.features.eval()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_images += batch_size

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    average_loss = total_loss / max(total_images, 1)
    accuracy = total_correct / max(total_images, 1)

    return (
        average_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# 9. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(model, class_to_idx, epoch, val_accuracy):
    checkpoint = {
        "architecture": "efficientnet_b0",
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": IMAGE_SIZE,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225]
    }

    torch.save(checkpoint, MODEL_PATH)

    print("\nModel checkpoint saved:")
    print(MODEL_PATH)


# ============================================================
# 10. EVALUATE THE TEST SET
# ============================================================

def evaluate_test_set(model, test_loader):
    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy, labels, predictions = run_epoch(
        model,
        test_loader,
        criterion
    )

    idx_to_class = {
        index: name
        for name, index in {"Fake": 0, "Real": 1}.items()
    }

    target_names = [
        idx_to_class[0],
        idx_to_class[1]
    ]

    print("\n" + "=" * 55)
    print("TEST RESULTS")
    print("=" * 55)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    print("\nClassification report:")

    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0
        )
    )

    print(
        "Confusion matrix "
        "(rows = actual, columns = predicted):"
    )

    print(
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        )
    )


# ============================================================
# 11. MAIN TRAINING FUNCTION
# ============================================================

def main():
    set_seed()

    print("=" * 55)
    print("DEEPFAKE DETECTION - EFFICIENTNET-B0")
    print("=" * 55)

    print("Project folder:", PROJECT_DIR)
    print("Dataset folder:", DATA_DIR)
    print("Model output:", MODEL_PATH)
    print("Device:", DEVICE)
    print("Smoke test:", SMOKE_TEST)

    if DEVICE.type == "cpu":
        print(
            "\nNOTE: PyTorch is using the CPU. "
            "Full-dataset training may take a long time."
        )

    (
        train_loader,
        val_loader,
        test_loader,
        class_to_idx
    ) = load_datasets()

    model = create_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    epochs_to_run = EPOCHS

    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(epochs_to_run):
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs_to_run}")

        train_loss, train_accuracy, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy, _, _ = run_epoch(
            model,
            val_loader,
            criterion
        )

        elapsed = time.time() - start_time

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy * 100:.2f}%"
        )

        print(f"Epoch time: {elapsed:.1f} seconds")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

            save_checkpoint(
                model,
                class_to_idx,
                epoch + 1,
                val_accuracy
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    evaluate_test_set(model, test_loader)

    print("\nTraining and evaluation completed.")
    print("Checkpoint:", MODEL_PATH)

    if SMOKE_TEST:
        print(
            "\nThis was a smoke test on a small subset. "
            "The reported test accuracy is not a reliable "
            "estimate of real-world performance."
        )
        print(
            "If the script runs successfully, you can later "
            "set SMOKE_TEST = False for full-dataset training."
        )


if __name__ == "__main__":
    main()
