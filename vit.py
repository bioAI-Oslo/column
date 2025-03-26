import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from transformers import ViTModel, ViTImageProcessor, ViTForImageClassification
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
import json

# Import damage methods
from zero_shot_damage import (
    sample_randomly,
    sample_squarely,
)


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        Made a small custom early stopping, which is more suitable for batch-level checks.
        """
        self.patience = patience
        self.delta = delta
        self.best_val_loss = np.Inf

        # This variable will be checked by prosesses outside the object
        self.stopped = False

        self.counter = 0

    def check(self, val_loss):
        """
        Checks whether the validation loss has improved. If it has, the counter is reset,
        otherwise the counter is increased. If the patience has been reached, the stopped
        flag is set to True and the function returns True. Otherwise, it returns False.

        Args:
            val_loss (float): The current validation loss.

        Returns
            bool: Whether the patience has been reached.
        """
        if self.stopped:
            return True

        # If a significant improvement has been made
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
        # if not, increase the counter
        else:
            self.counter += 1

        # Has out patience been reached?
        self.stopped = self.counter >= self.patience
        return self.stopped


def train(model, train_loader, validation_loader, epochs=3, plotting=False):
    """
    Train a model on given data.

    Args:
        model (nn.Module): A PyTorch model to be trained.
        train_loader (DataLoader): A PyTorch DataLoader containing the training data.
        validation_loader (DataLoader): A PyTorch DataLoader containing the validation data.
        epochs (int, optional): The number of epochs to run the training loop. Defaults to 3.
        plotting (bool, optional): Whether to plot the training and validation loss/accuracy. Defaults to False.

    Returns:
        nn.Module: The trained model.
    """
    model.train()

    # Define loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Recording losses
    train_losses = []
    val_losses = []
    val_accuracies = []
    epochs_recorded = []
    val_loss_best = np.inf

    # Our early stopping will be applied to batch-level checks
    early_stopping = EarlyStopping(patience=7, delta=0.005)
    check_interval = 20

    # This try-except is for being able to interrupt the training with ctrl+c under development
    try:
        steps = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epochs_recorded.append(epoch)
            batch = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                print(f"Batch {batch}: Loss is {loss.item()}")

                if steps % check_interval == 0:
                    # Record accuracies
                    val_acc_now, val_loss_now = evaluate(model, validation_loader, loss_function)
                    print(f"Validation loss: {val_loss_now}, Validation accuracy: {val_acc_now}")

                    # Record run metrics
                    train_losses.append(loss.item())
                    val_losses.append(val_loss_now)
                    val_accuracies.append(val_acc_now)

                    # Has a significant improvement been made? If yes, save model
                    if val_loss_now < val_loss_best - early_stopping.delta:
                        val_loss_best = val_loss_now
                        save_model(model, superfolder)

                    if early_stopping.check(val_loss_now):
                        print("Early stopping triggered")
                        break
                    model.train()  # Reset model to train mode

                batch += 1
                steps += 1

            if early_stopping.stopped:
                print("Early stopping has been activated. Moving on...")
                break

    except KeyboardInterrupt:
        print("Training interrupted. Moving on...")

    if plotting:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xticks(np.arange(len(train_losses)), check_interval * np.arange(len(train_losses)))
        plt.xlabel("Train step")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.array(val_accuracies) * 100, label="Validation Accuracy")
        plt.xticks(np.arange(len(val_accuracies)), check_interval * np.arange(len(val_accuracies)))
        plt.xlabel("Train step")
        plt.ylabel("Accuracy (%)")
        plt.show()

    # Save plotting data with json
    data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "train_steps": [i * check_interval for i in range(len(train_losses))],
        "epochs_recorded": epochs_recorded,
    }

    with open(os.path.join(run_folder, "plotting_data"), "w") as f:
        json.dump(data, f)

    return model


def visualize_run(load_name):
    """
    Visualize the results of a run.

    Args:
        load_name (str): The name of the json file to load.
    """
    with open(load_name, "r") as f:
        data = json.load(f)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(data["train_losses"], label="Train Loss")
    plt.plot(data["val_losses"], label="Validation Loss")
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.array(data["val_accuracies"]) * 100, label="Validation Accuracy")
    plt.xlabel("Train step")
    plt.ylabel("Accuracy (%)")
    plt.show()


def get_data(data_name="mnist", batch_size=32, validation_size=0.02, classes=[0, 1, 2, 3, 4]):
    """
    Loads and preprocesses the specified dataset, returning DataLoader objects for training, validation, and testing.

    Args:
        data_name (str, optional): The name of the dataset to load. Options are "mnist", "fashionmnist", or "cifar". Defaults to "mnist".
        batch_size (int, optional): The size of the batches to use in the DataLoader. Defaults to 32.
        validation_size (float, optional): The proportion of the training data to use for validation. Defaults to 0.02.
        classes (list, optional): A list of class labels to include in the dataset. Defaults to [0, 1, 2, 3, 4].

    Returns:
        tuple: A tuple containing three DataLoader objects for the training, validation, and test datasets, respectively.
    """

    # Using the recommended preprocessing (actually no, I struggled with input not being as required by the model)
    # img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", do_convert_rgb=True)

    # Found the default treatment from ViTImageProcessor, using that with transforms

    if data_name == "cifar":
        print("Using CIFAR10 data")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Using IMAGENET_STANDARD_MEAN and IMAGENET_STANDARD_STD
            ]
        )
    else:
        print("Using MNIST/FASHION data")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Using IMAGENET_STANDARD_MEAN and IMAGENET_STANDARD_STD
            ]
        )

    if data_name == "mnist":
        train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    elif data_name == "fashionmnist":
        train_dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = FashionMNIST(root="./data", train=False, transform=transform, download=True)

    elif data_name == "cifar":
        train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)

    # Remove unwanted classes
    def get_filtered_indexes(dataset, classes):
        # Get indexes that correspond to the classes
        idx = [i for i in range(len(dataset)) if dataset.targets[i] in classes]
        return idx

    # Set aside some indexes for a validation set
    train_indexes = get_filtered_indexes(train_dataset, classes)
    val_indexes = train_indexes[: int(len(train_indexes) * validation_size)]
    train_indexes = train_indexes[int(len(train_indexes) * validation_size) :]

    # Get the train and validation datasets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)
    validation_dataset = torch.utils.data.Subset(train_dataset, val_indexes)

    # Get the test dataset
    test_dataset = torch.utils.data.Subset(test_dataset, get_filtered_indexes(test_dataset, classes))

    # Load up dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


def test_damage_single(model, batch, test_sizes_percentages, sample_function=sample_squarely):
    """
    Tests the performance of a model when damaging its embeddings.

    Args:
        model (nn.Module): The model to test.
        batch (tuple): A tuple of images and labels.
        test_sizes_percentages (list): A list of percentages [0, 1] to test the model at.
        sample_function (callable): A function to sample a set of patches to silence. Defaults to sample_squarely.

    Returns:
        list: A list of the accuracy of the model at each test size percentage.
    """
    model.eval()

    with torch.no_grad():
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Record accuracies
        accuracies = []

        for test_size in test_sizes_percentages:
            # The patch embedding block consists of a linear projection and optionally dropout
            out_embed = model.vit.embeddings(images)  # [B, num_patches, num_features]

            B, num_patches, _ = out_embed.shape
            # Getting indexes to silence from the pre-made methods
            patches_one_dim = int(np.sqrt(num_patches - 1))
            for b in range(B):
                # Getting indexes to silence
                indexes = sample_function(
                    int(np.floor(test_size * (out_embed.shape[1] - 1))),
                    patches_one_dim,
                    patches_one_dim,
                )

                # To silence some patches, I should silence all features of the ith patch [B,i,:]
                for x, y in zip(indexes[0], indexes[1]):
                    index = (x - 1) * patches_one_dim + (y - 1) + 1
                    out_embed[b, index, :] = 0

            # The encoder block consists of 12 attention ViT blocks. It is a BaseModelOutput object
            out_encoder = model.vit.encoder(out_embed, output_attentions=True)
            # Retrieving the last_hidden_state from the BaseModelOutput object and sending it through the norm layer
            out_layer_norm = model.vit.layernorm(out_encoder.last_hidden_state)
            # We wish to only use the linear classifier on the CLS token, which is at position 0. Otherwise, we use all batch-elements and info
            logits = model.classifier(out_layer_norm[:, 0, :])

            predicted = torch.argmax(logits, dim=1)
            batch_size = labels.shape[0]

            accuracy = (predicted == labels).sum().item() / batch_size
            accuracies.append(accuracy)

            print("At test size", test_size, "accuracy is now", accuracy, "%")

            # Visualizing attention for damage
            """only_first_attention = [vit_layer[0, :, :, :] for vit_layer in out_encoder.attentions]
            print(only_first_attention[0].shape, len(only_first_attention))

            visualize_rollout(
                [vit_layer[0:1, :, :, :] for vit_layer in out_encoder.attentions],
                images[0].permute(1, 2, 0).cpu().detach().numpy(),
                labels[0],
                predicted,
            )
            plt.show()"""

    return accuracies


def get_balanced_set(loader, classes, samples_per_class=40):
    """
    Collects a balanced dataset of images from the given loader.
    The returned dataset consists of 'samples_per_class' images from each class specified in 'classes'.
    The images are returned as a tensor of shape (samples_per_class * len(classes), 3, 224, 224)
    and the labels as a tensor of shape (samples_per_class * len(classes)).

    Args:
        loader (torch.utils.data.DataLoader): The iterable of (image, label) tuples
        classes (list): A list or tuple of the classes to collect
        samples_per_class (int): The number of samples to collect from each class, defaults to 40

    Returns:
        A tuple of two tensors, the first containing the balanced set of images and the second containing the labels
    """
    # Creating empty tensors
    balanced_set_images = torch.zeros((len(classes) * samples_per_class, 3, 224, 224))
    balanced_set_labels = torch.zeros(len(classes) * samples_per_class)

    # The running index
    index = 0

    for c in classes:
        nr_collected = 0  # Nr collected per class so far

        # Going through batches
        for image, label in loader.dataset:
            # Is this our current class? If yes, we collect it
            if label == c and nr_collected < samples_per_class:
                balanced_set_images[index] = image
                balanced_set_labels[index] = label
                index += 1
                nr_collected += 1

            # Did we collect enough images for this class? If yes, continue to next class
            if nr_collected == samples_per_class:
                break

    # No need to shuffle, cause we will go through all images
    return balanced_set_images, balanced_set_labels


def test_damage(superfolder, test_loader, classes, sample_function=sample_randomly):
    """
    Test the robustness of all models in a folder to different test sizes.

    The function will go through all folders in the superfolder, and for each folder, it will
    test the robustness of the model to different test sizes. The test size is given as a
    percentage of the total embedding size, from 0 to 100%. The accuracy of the model is
    calculated at each test size, and the average accuracy and standard deviation are plotted
    afterwards.

    Args:
        superfolder (str): The superfolder containing all folders with models to test
        test_loader (torch.utils.data.DataLoader): The iterable of (image, label) tuples
        classes (list): A list of the classes to test
        sample_function (function, optional): The function to use for sampling the test set. Defaults to sample_randomly

    Returns:
        None
    """
    # To record performances for every folder
    folder_performances = {}

    # Construct path
    path = "./experiments/vit/" + superfolder + "/"

    # Every network will try their hand at this data
    batch = get_balanced_set(test_loader, classes)

    # Get test sizes
    test_sizes_percentages = np.linspace(0, 1, 11)

    # Go through every folder in superfolder
    for folder in os.listdir(path):
        # Check if it is a folder
        if os.path.isdir(path + folder):
            # Load model
            model = load_model(path + folder + "/checkpoint_model.pth", classes=classes)
            accuracies = test_damage_single(model, batch, test_sizes_percentages, sample_function)

            folder_performances[folder] = accuracies

    # Plot
    for folder, accuracies in folder_performances.items():
        plt.plot(100 * np.array(accuracies), linestyle="--")

    plt.plot(100 * np.mean(list(folder_performances.values()), axis=0), color="k", linewidth=3)

    # Plot the standard deviation
    plt.fill_between(
        np.linspace(0, 10, 11),
        100 * np.mean(list(folder_performances.values()), axis=0)
        - 100 * np.std(list(folder_performances.values()), axis=0),
        100 * np.mean(list(folder_performances.values()), axis=0)
        + 100 * np.std(list(folder_performances.values()), axis=0),
        alpha=0.2,
    )

    plt.xticks(np.linspace(0, 10, 11), np.linspace(0, 100, 11, dtype=int))
    plt.yticks(np.linspace(0, 100, 11), np.linspace(0, 100, 11, dtype=int))
    plt.xlabel("Test size (% of total embedding)")
    plt.ylabel("Accuracy (%)")
    plt.title(sample_function.__name__.replace("_", " ").capitalize())
    plt.show()

    # Save dictionary
    folder_performances["test_sizes"] = list(test_sizes_percentages)
    json.dump(
        folder_performances,
        open(
            path + f"{sample_function.__name__.replace('sample_', '').replace('ly', ' ')}_silencing_robustness.json",
            "w",
        ),
    )


def test_equivalence(model, test_loader):
    """
    Compare the evaluation results of a model using two different approaches:
    the normal full model pass and a gradual pass through the model's components.

    This function is kept as documentation that the approach used in test_damage_single works.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): A DataLoader containing the test dataset.
    """
    model.eval()

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)

        # Normal way
        outputs = model(images)
        predicted = torch.argmax(outputs.logits, dim=1)
        batch_size = labels.shape[0]

        # Record normal accuracy
        accuracy_normal = (predicted == labels).sum().item() * 100 / batch_size
        sum_normal = np.sum(outputs.logits.cpu().detach().numpy())

        # Now trying to do it gradually

        # The patch embedding block consists of a linear projection and optionally dropout
        out_embed = model.vit.embeddings(images)
        # The encoder block consists of 12 attention ViT blocks. It is a BaseModelOutput object
        out_encoder = model.vit.encoder(out_embed)
        # Retrieving the last_hidden_state from the BaseModelOutput object and sending it through the norm layer
        out_layer_norm = model.vit.layernorm(out_encoder.last_hidden_state)
        # We wish to only use the linear classifier on the CLS token, which is at position 0. Otherwise, we use all batch-elements and features
        logits = model.classifier(out_layer_norm[:, 0, :])

        predicted = torch.argmax(logits, dim=1)
        batch_size = labels.shape[0]

        # Record gradual accuracy
        accuracy_gradual = (predicted == labels).sum().item() * 100 / batch_size
        sum_gradual = np.sum(logits.cpu().detach().numpy())

        # Print comparison
        print("Sums of logits for both systems:")
        print("Normal:", sum_normal, "Gradual:", sum_gradual)
        print("Accuracy for both systems:")
        print("Normal:", accuracy_normal, "Gradual:", accuracy_gradual)


def evaluate(model, loader, loss_fn=None, batches=None):
    """
    Evaluate the model on a given DataLoader.
    If loss_fn is not given, it will not calculate the loss.
    If batches is not gievn, it will evaluate on all batches.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): A DataLoader containing the dataset to be evaluated on.
        loss_fn (nn.Module, optional): A loss function used to calculate the loss. Defaults to None.
        batches (int, optional): The number of batches to evaluate on. Defaults to None.

    Returns:
        float: The accuracy of the model on the given dataset.
        float, optional: The average loss of the model on the given dataset.
    """
    model.eval()

    correct = 0
    loss_sum = 0
    total_nr_batches = 0
    total_nr_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()

            # Calculate loss, optionally
            if loss_fn is not None:
                loss = loss_fn(outputs.logits, labels)
                loss_sum += loss.item()

            total_nr_samples += len(labels)
            total_nr_batches += 1

            # Break if batches is not None and reached the max number of batches
            if batches is not None and total_nr_batches >= batches:
                break

    accuracy = correct / total_nr_samples
    if loss_fn is not None:
        return accuracy, loss_sum / total_nr_batches
    return accuracy


def save_model(model, superfolder):
    """
    Saves the model parameters in a file called "checkpoint_model.pth" in a folder uniquely named by the current time and date.

    When this function is run, the script may or may not have already created the experiment folder.
    If it has not, it will create it.

    Args:
        model (nn.Module): The model to be saved.
        superfolder (str): The name of the experiment folder.
    """
    global run_folder
    if run_folder is None:
        # Make a unique experiment folder name by the time and date
        name = f"{time.localtime().tm_mday}-{time.localtime().tm_mon}-" + str(time.localtime().tm_year)[-2:]
        name += f"_{time.localtime().tm_hour}:{time.localtime().tm_min}"

        # Sometimes, the path is already made (but not in this script). Add a unique numerical suffix
        additive = 2
        new_name = name
        while os.path.isdir(f"./experiments/vit/{superfolder}/{new_name}"):
            new_name = name + "_" + str(additive)
            additive += 1

        run_folder = f"./experiments/vit/{superfolder}/{new_name}"

        os.mkdir(run_folder)

    file_path = f"{run_folder}/checkpoint_model.pth"

    # Save model parameters
    print("Saving model...")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path, classes=[0, 1, 2]):
    """
    Loads the model parameters from a file and returns a ViT model.

    Args:
        file_path (str): The path to the model parameters file.
        classes (list, optional): The list of classes to be used in the model head. Defaults to [0, 1, 2].

    Returns:
        nn.Module: A ViT model ready for evaluation.
    """
    # Load a pre-trained ViT model
    model = new_model(classes)
    # Furnish it with pre-trained weights (the ones I trained)
    model.load_state_dict(torch.load(file_path))

    return model


def new_model(classes=[0, 1, 2]):
    """
    Creates a new ViT model.

    Args:
        classes (list, optional): The list of classes to be used in the model head. Defaults to [0, 1, 2].

    Returns:
        nn.Module: A new ViT model ready for evaluation.
    """
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name, num_labels=len(classes), output_attentions=True, attn_implementation="eager"
    )
    model.to(device)

    return model


def normalize_to_0_and_1(matrix):
    """
    Normalize the given matrix to the range [0, 1] by subtracting the minimum value and dividing by the range.

    Args:
        matrix (np.ndarray): The matrix to be normalized.

    Returns:
        np.ndarray: The normalized matrix.
    """
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))


def visualize_attn_layer(layer_attn, img, label, belief):
    """
    Visualizes the attention weights of a single layer in the model.

    The resulting plot will have three subplots:
    1. The original image
    2. The heatmap of attention weights
    3. The superposition of the image and heatmap

    The heatmap is resized to the same size as the image. The attention weights are normalized to the range [0, 1] before plotting.

    Args:
        layer_attn (torch.Tensor): The attention weights of a single layer in the model, of shape (heads, num_patches, num_patches).
        img (torch.Tensor): The input image, of shape (channels, height, width).
        label (torch.Tensor): The label of the input image, of shape ().
        belief (int): The believed class of the model, of shape ().
    """
    attn_head = layer_attn.mean(axis=1)[0]  # Get the mean across heads

    # Remove CLS token, average over all heads, and reshape to 14x14 grid
    attn_patches = attn_head[1:, 1:].mean(dim=0).reshape(14, 14)  # How mcuh each patch is weighted by other patches
    # attn_patches = attn_head[0, 1:].reshape(14, 14)  # How much each patch is weighted by CLS token
    attn_to_plot = attn_patches.cpu().detach().numpy()

    # Resizing attn_to_plot
    attn_to_plot_resized = np.array(
        cv2.resize(attn_to_plot, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)
    )

    cmap = "viridis"
    fig = plt.figure(frameon=False)
    fig.suptitle("Believed class is " + str(belief) + ", correct class is " + str(int(label)))

    # Plotting the image
    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.title("image")
    plt.xticks([])
    plt.yticks([])

    # Plotting the heatmap
    plt.subplot(132)
    plt.imshow(attn_to_plot, cmap=cmap)
    plt.title("heatmap")
    plt.xticks([])
    plt.yticks([])

    # Plotting the superposition
    plt.subplot(133)
    plt.imshow(
        0.7 * normalize_to_0_and_1(img),
        vmin=0,
        vmax=1,
        cmap="gray",
    )  # "gray" is just ignored for color images
    plt.imshow(
        normalize_to_0_and_1(attn_to_plot_resized),
        alpha=normalize_to_0_and_1(attn_to_plot_resized),
        cmap=cmap,
    )
    plt.title("image + heatmap")
    plt.xticks([])
    plt.yticks([])


def visualize_rollout(attn_maps, img, label, belief):
    """
    Visualizes the attention rollout from multiple layers of a vision transformer model.

    This function computes the attention rollout by iteratively multiplying attention matrices
    from multiple layers. The attention rollout indicates how much each patch is influenced by
    the [CLS] token, normalized for visualization. It then plots the original image, the attention
    heatmap, and their superposition.

    Args:
        attn_maps (list of torch.Tensor): List of attention maps from different layers, each of shape
                                          (batch_size, heads, num_patches, num_patches).
        img (np.ndarray): The input image to display, typically of shape (height, width) or
                          (height, width, channels).
        label (int): The true label of the input image.
        belief (int): The predicted label of the model for the input image.
    """

    # Start with identity
    num_patches = attn_maps[0].shape[-1]
    rollout = torch.eye(num_patches).to(device)

    L = len(attn_maps)
    # Let's go backwards and propagate the attention
    for l in range(L - 1, -1, -1):
        # Take the mean across layer l, over all heads
        mean_across_heads = attn_maps[l][0].mean(dim=0)  # Shape (num_patches x num_patches)

        # Add identity to stabilize
        mean_across_heads += torch.eye(num_patches).to(device)

        rollout = rollout @ mean_across_heads

    # How mcuh each patch is weighted by CLS token
    attn_patches = rollout[0, 1:].reshape(14, 14)

    # Final normalization for visualization
    attn_patches = (attn_patches - torch.min(attn_patches)) / (torch.max(attn_patches) - torch.min(attn_patches))

    # Resizing attn_patches to image dimensions
    normalized_rollout_resized = np.array(
        cv2.resize(attn_patches.cpu().detach().numpy(), (img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)
    )

    cmap = "viridis"
    fig = plt.figure(frameon=False)
    fig.suptitle("Believed class is " + str(belief) + ", correct class is " + str(int(label)))

    # Plotting the image
    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.title("image")
    plt.xticks([])
    plt.yticks([])

    # Plotting the heatmap
    plt.subplot(132)
    plt.imshow(normalized_rollout_resized, cmap=cmap)
    plt.title("heatmap")
    plt.xticks([])
    plt.yticks([])

    # Plotting the superposition
    plt.subplot(133)
    plt.imshow(
        0.7 * normalize_to_0_and_1(img),
        vmin=0,
        vmax=1,
        cmap="gray",
    )  # "gray" is just ignored for color images
    plt.imshow(
        normalized_rollout_resized,
        alpha=normalized_rollout_resized,
        cmap=cmap,
    )
    plt.title("image + heatmap")
    plt.xticks([])
    plt.yticks([])


def plot_attn_maps(test_loader, num_images=5, only_last=False, rollout=True):
    """
    Plots attention maps of the given test_loader.

    Args:
        test_loader (torch.utils.data.DataLoader): The data loader containing the test data.
        num_images (int, optional): The number of images to plot from the test_loader. Defaults to 5.
        only_last (bool, optional): Whether to only plot the last attention map. Defaults to False. Not important if rollout is True.
        rollout (bool, optional): Whether to plot the attention rollout or the attention maps. Defaults to True.
    """
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            for img, label in zip(batch[0][:num_images], batch[1][:num_images]):
                img_tensor = img.to(device)

                img_tensor = img_tensor.unsqueeze(0)

                if img_tensor.shape[1] == 1:  # if grayscale
                    img_to_plot = img_tensor.cpu().detach().numpy()[0, 0]
                else:
                    img_to_plot = img_tensor[0].permute(1, 2, 0).cpu().detach().numpy()

                # Normalize img_to_plot
                img_to_plot = (img_to_plot - np.min(img_to_plot)) / (np.max(img_to_plot) - np.min(img_to_plot))

                # Get attention maps
                outputs = model(img_tensor)
                belief = np.argmax(outputs.logits.cpu().detach().numpy())
                attn_maps = outputs.attentions  # Already extracted for you!

                print(attn_maps[0].shape, len(attn_maps))

                if rollout:
                    visualize_rollout(attn_maps, img_to_plot, label, belief)
                else:

                    if only_last:
                        visualize_attn_layer(attn_maps[-1], img_to_plot, label, belief)
                    else:
                        for layer_attn in attn_maps:
                            visualize_attn_layer(layer_attn, img_to_plot, label, belief)
                plt.show()
            break


def test_performance(superfolder, train_loader, validation_loader, test_loader):
    """
    Tests the performance of all models in a superfolder.

    Args:
        superfolder (str): The superfolder containing all folders with models to test
        train_loader (torch.utils.data.DataLoader): The iterable of (image, label) tuples for the training data
        validation_loader (torch.utils.data.DataLoader): The iterable of (image, label) tuples for the validation data
        test_loader (torch.utils.data.DataLoader): The iterable of (image, label) tuples for the test data
    """
    folder_performances = {}

    # Construct path
    path = "./experiments/vit/" + superfolder + "/"

    # Go through every folder in superfolder
    for folder in os.listdir(path):
        # Check if it is a folder
        if os.path.isdir(path + folder):
            # Load model
            model = load_model(path + folder + "/checkpoint_model.pth", classes=classes)

            # Collect accuracies
            accuracy_train = evaluate(model, train_loader, batches=None)
            accuracy_val = evaluate(model, validation_loader, batches=None)
            accuracy_test = evaluate(model, test_loader, batches=None)

            folder_performances[folder] = {"train": accuracy_train, "validation": accuracy_val, "test": accuracy_test}

    best_val_acc_folder = list(folder_performances.keys())[0]
    best_val_acc_score = folder_performances[best_val_acc_folder]["validation"]

    # Recording mean and std as [mean, std]
    train_stats = []
    validation_stats = []
    test_stats = []

    for folder, accuracies in folder_performances.items():
        if accuracies["validation"] > best_val_acc_score:
            best_val_acc_score = accuracies["validation"]
            best_val_acc_folder = folder

        train_stats.append(accuracies["train"])
        validation_stats.append(accuracies["validation"])
        test_stats.append(accuracies["test"])

    train_mean, train_std = np.mean(train_stats), np.std(train_stats)
    validation_mean, validation_std = np.mean(validation_stats), np.std(validation_stats)
    test_mean, test_std = np.mean(test_stats), np.std(test_stats)

    print(
        f"""
            Performance:
            Best performance: {best_val_acc_folder}, with 
            train: {folder_performances[best_val_acc_folder]["train"]} validation accuracy: {best_val_acc_score} test: {folder_performances[best_val_acc_folder]["test"]}
            Train mean accuracy: {train_mean} std: {train_std}
            Validation mean accuracy: {validation_mean} std: {validation_std}
            Test mean accuracy: {test_mean} std: {test_std}

        """
    )


if __name__ == "__main__":
    # Below are parameters that can be changed by the user
    pretrained_model_name = "google/vit-base-patch16-224-in21k"
    device = "mps"  # or "cpu", on Mac M1 "mps" is the best choice, "cuda" is best on Linux
    data_name = "mnist"
    classes = [0, 1, 2, 3, 4]
    superfolder = "mnist5"
    run_folder = None

    # load_name = "./experiments/vit/mnist5/18-3-25_14:35"

    # What function should the script do?
    train_now = False
    load_now = False
    test_damage_now = False
    test_performance_now = True
    evaluate_now = False
    visualize_attention = False
    print_model_now = False
    visualize_run_now = False

    train_loader, validation_loader, test_loader = get_data(data_name, classes=classes)

    if train_now:
        model = new_model(classes=classes)
        model = train(model, train_loader, validation_loader, plotting=False)

    if load_now:
        model = load_model(load_name + "/checkpoint_model.pth", classes=classes)

    if visualize_run_now:
        visualize_run(load_name + "/plotting_data")

    if test_damage_now:
        test_damage(superfolder, test_loader, classes, sample_function=sample_squarely)

    if test_performance_now:
        test_performance(superfolder, train_loader, validation_loader, test_loader)

    if evaluate_now:
        accuracy = evaluate(model, test_loader)
        print(f"Accuracy: {100 * accuracy}%")

    if visualize_attention:
        plot_attn_maps(test_loader, num_images=10, only_last=False, rollout=True)

    if print_model_now:
        print(model)  # Shows layers and info
