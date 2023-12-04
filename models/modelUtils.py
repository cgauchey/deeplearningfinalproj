import numpy as np
import torch
from utils import constants
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F 
import os
import datetime

def regression_loss(output, target, num_classes, alpha=0.9):
    # Assumes that output is in the shape (batch_size, [num_classes + 6])
    # These values are the class logits, pitch, roll, yaw, x, y, z values
    # Assumes that target is in the shape (batch_size, [1 + 6])
    # These values are the class as int, pitch, roll, yaw, x, y, z values

    print("num classes: ", num_classes)

    print("output: ", output.shape)
    print("target: ", target.shape)

    # Split the class from the 6dof pose values (first num classes are the class logits)
    output_class, output_pose = output[:, :num_classes], output[:, num_classes:]
    target_class, target_pose = target[:, :1], target[:, 1:]

    print("output_class: ", output_class.shape)
    print("output_pose: ", output_pose.shape)
    print("target_class: ", target_class.shape)
    print("target_pose: ", target_pose.shape)

    # Use cross entropy loss for the class
    class_loss = F.cross_entropy(output_class, target_class.squeeze().long()).float()

    # Use 6D euclidean distance as the pose loss
    pose_loss = F.mse_loss(output_pose, target_pose).float()

    # Combine them with the alpha values
    total_loss = alpha * class_loss + (1 - alpha) * pose_loss
    print(type(total_loss))

    return total_loss

def make_inference(model, image):
    # Set the model to evaluation mode
    model.eval()

    # Move the image to the device
    image = image.to(model.device)

    # Forward pass
    output = model(image)

    # First, split the outputs into the class logits and the 6dof pose values
    class_logits, pose_values = output[:, :model.num_classes], output[:, model.num_classes:]

    # Get the class predictions with the softmax
    class_preds = model.softmax(class_logits)

    return class_logits, class_preds, pose_values


def train(model, optimizer, train_dataset, val_dataset, epochs=20, batch_size=32, patience=5, 
          seed=42, print_freq=5, save_freq=10, model_save_folder=None, verbose=False):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Make output folder if it doesn't exist
    if model_save_folder is not None and not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    best_model = None
    best_epoch_num = 0

    if verbose:
        print("Starting training on device: {}".format(model.device))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Make list to store the training losses and validation losses
    train_losses = []
    val_losses = []

    epoch_train_loss = 0

    # Loop over the epochs
    for epoch in range(epochs):

        # Set the model to training mode
        model.train()

        if verbose and epoch+1 % print_freq == 0:
            print("Epoch {}/{}".format(epoch+1, epochs))
        
        # Loop over the batches
        for i, (X, y) in enumerate(train_loader):

            # Zero out the gradients
            optimizer.zero_grad()
            
            # Move the data to the device
            X = X.to(model.device)
            y = y.to(model.device)

            # Forward pass
            output = model(X)

            # Calculate the loss
            loss = regression_loss(output, y, model.num_classes)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            epoch_train_loss += loss.item()

        # Evaluation
        model.eval()

        with torch.no_grad():

            # Compute the val loss in batches
            epoch_val_loss = 0
            for i, (X, y) in enumerate(val_loader):

                # Move the data to device 
                X = X.to(model.device)
                y = y.to(model.device)

                # Forward pass
                output = model(X)

                # Calculate the loss
                loss = regression_loss(output, y, model.num_classes)

                epoch_val_loss += loss.item()

        # Average the losses
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)

        # Save the losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if verbose and epoch+1 % print_freq == 0:
            print("Epoch: {} - train loss: {}".format(epoch+1, epoch_train_loss))
            print("Epoch: {} - val loss: {}".format(epoch+1, epoch_val_loss))
        
        # Check if this is the best model
        if epoch_val_loss < min(val_losses):
            best_model = model
            best_epoch_num = epoch
        
        # Check if we should save the model
        if model_save_folder is not None and epoch+1 % save_freq == 0:
            # get a timestamp for the name
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_name = timestamp + "_epoch_{}".format(epoch+1)
            torch.save(model.state_dict(), os.path.join(model_save_folder, save_name))
        
        # Check if we should stop early
        if epoch > patience:
            if val_losses[-1] >= max(val_losses[-patience:]):
                if verbose:
                    print("Validation loss has not improved in {} epochs, stopping training at epoch {}".format(
                        patience, epoch+1))
                break
        
    if verbose:
        print("Training finished. Best validation loss of {} at epoch {}".format(min(val_losses), best_epoch_num+1))
    
    # Save the final model if path given
    if model_save_folder is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_name = timestamp + "final_epoch_{}".format(best_epoch_num+1)
        torch.save(best_model.state_dict(), os.path.join(model_save_folder, save_name))

    return best_model, best_epoch_num, train_losses, val_losses


def evaluate_model(model, test_dataset, batch_size=32, verbose=False):

    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Set the model to evaluation mode
    model.eval()

    # Make lists to store the losses
    losses = []

    # Loop over the batches
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):

            # Move the data to device
            X = X.to(model.device)
            y = y.to(model.device)

            # Forward pass
            output = model(X)

            # Calculate the loss
            loss = regression_loss(output, y, model.num_classes)

            losses.append(loss.item())
    
    # Average the losses
    avg_loss = sum(losses) / len(losses)

    if verbose:
        print("Average loss: {}".format(avg_loss))
    
    return avg_loss
