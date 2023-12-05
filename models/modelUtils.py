import numpy as np
import torch
from utils import constants
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F 
import os
import datetime
import torch
from matplotlib import pyplot as plt

def compute_loss(output, target, num_classes, alpha=0.9):
    # Assumes that output is in the shape (batch_size, [num_classes + 6])
    # These values are the class logits, pitch, roll, yaw, x, y, z values
    # Assumes that target is in the shape (batch_size, [1 + 6])
    # These values are the class as int, pitch, roll, yaw, x, y, z values

    # Split the class from the 6dof pose values (first num classes are the class logits)
    output_class, output_pose = output[:, :num_classes], output[:, num_classes:]
    target_class, target_pose = target[:, :1], target[:, 1:]

    # Use cross entropy loss for the class
    class_loss = F.cross_entropy(output_class, target_class.squeeze().long()).float()

    # Use 6D euclidean distance as the pose loss
    pose_loss = F.mse_loss(output_pose, target_pose).float()

    # Combine them with the alpha values
    total_loss = alpha * class_loss + (1 - alpha) * pose_loss

    return total_loss

def make_inference(model, image):
    # Set the model to evaluation mode
    model.eval()

    # Move the image to the device
    image = image.to(model.device).float()

    # Forward pass
    output = model(image)

    # First, split the outputs into the class logits and the 6dof pose values
    class_logits, pose_values = output[:, :model.num_classes], output[:, model.num_classes:]

    # Get the class predictions with the softmax
    class_preds = model.softmax(class_logits)

    return class_logits, class_preds, pose_values


def plot_random_images(model, dataset, save_folder, num_images=10):
    # Pick 10 random images from the dataset
    indices = np.random.choice(len(dataset), num_images, replace=False)

    # Get the images and labels
    images = [dataset[i][0] for i in indices]
    labels = [dataset[i][1] for i in indices]

    # Remember labels are in the shape (1 + 6)
    # First value is the class, then the 6dof pose values
    label_classes = [label[0] for label in labels]
    label_poses = [label[1:] for label in labels]

    # Make the predictions
    class_logits, class_preds, pose_values = make_inference(model, torch.stack(images))

    # Get the class predictions
    class_preds = torch.argmax(class_preds, dim=1)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 2, i+1)
        plt.imshow(images[i].permute(1, 2, 0))

        # Get the predicted and actual class
        pred_class = class_preds[i].item()
        actual_class = label_classes[i].item()

        # Get the predicted and actual pose values
        pred_pose = pose_values[i]
        actual_pose = label_poses[i]

        # Make a title that shows the predicted classes and posses
        title = "Predicted class: {}\nActual class: {}\n\n".format(pred_class, actual_class)
        title += "Predicted pose: {}\nActual pose: {}".format(pred_pose, actual_pose)
        plt.title(title)
        plt.axis('off')

    # Make the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Make a name with timestamp
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Save the plot
    plt.savefig(os.path.join(save_folder, timestamp + "_prediction_examples.png"))
    


def train(model, optimizer, train_dataset, val_dataset, epochs=20, batch_size=32, patience=5, 
          seed=42, print_freq=5, save_freq=10, model_save_folder=None, verbose=False, logfolder=None):
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Make output folder if it doesn't exist
    if model_save_folder is not None and not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    if logfolder is not None:
        # Make logfolder if it doesn't exist
        if not os.path.exists(logfolder):
            os.makedirs(logfolder)
        
        # Make a new logfile with the current timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        logfile = os.path.join(logfolder, timestamp + "_training.txt")
        logfile_fp = open(logfile, 'w')
        logfile_fp.write("Training log for model at timestamp: {}\n".format(timestamp))

    else:
        logfile_fp = None

    best_model = None
    best_epoch_num = 0

    if verbose:
        print(f"Starting training on device: {model.device}, device 0 is: {torch.cuda.get_device_name(0)}")

    if logfile_fp is not None:
        logfile_fp.write(f"Starting training on device: {model.device}, device 0 is: {torch.cuda.get_device_name(0)}\n")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Make list to store the training losses and validation losses
    train_losses = []
    val_losses = []

    epoch_train_loss = 0

    # Loop over the epochs
    for epoch in range(epochs):

        # Set the model to training mode
        model.train()
        
        # Loop over the batches
        for i, (X, y) in enumerate(train_loader):

            # Zero out the gradients
            optimizer.zero_grad()
            
            # Move the data to the device
            X = X.to(model.device).float()
            y = y.to(model.device).float()

            # Forward pass
            output = model(X)

            # Calculate the loss
            loss = compute_loss(output, y, model.num_classes)

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
                X = X.to(model.device).float()
                y = y.to(model.device).float()

                # Forward pass
                output = model(X)

                # Calculate the loss
                loss = compute_loss(output, y, model.num_classes)

                epoch_val_loss += loss.item()

        # Average the losses
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)

        # Save the losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if ((epoch+1) % print_freq) == 0:
            if verbose:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print("{}\tEpoch: {} - train loss:\t{}".format(timestamp, epoch+1, epoch_train_loss))
                print("{}\tEpoch: {} - val loss:\t{}\n".format(timestamp, epoch+1, epoch_val_loss))

            if logfile_fp is not None: 
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                logfile_fp.write("{}\tEpoch: {} - train loss:\t{}\n".format(timestamp, epoch+1, epoch_train_loss))
                logfile_fp.write("{}\tEpoch: {} - val loss:\t{}\n\n".format(timestamp, epoch+1, epoch_val_loss))

        # Check if this is the best model
        if best_model is None or epoch_val_loss == min(val_losses):
            best_model = model
            best_epoch_num = epoch
            if verbose:
                print("Updating best model, now found at epoch {}".format(epoch+1))
            
            if logfile_fp is not None:
                logfile_fp.write("Updating best model, now found at epoch {}\n".format(epoch+1))

        # Check if we should save the model
        if model_save_folder is not None and (epoch+1) % save_freq == 0:
            # get a timestamp for the name
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_name = timestamp + "_epoch_{}".format(epoch+1)
            torch.save(model.state_dict(), os.path.join(model_save_folder, save_name))
            if verbose:
                print("Saving model weights at epoch {}".format(epoch+1))
            
            if logfile_fp is not None:
                logfile_fp.write("Saving model weights at epoch {}\n".format(epoch+1))

        # Check if we should stop early
        # Stop early if after patience window and val loss is worse than all previous losses in window
        # using ">" because this epoch loss is in the window and we want to stop if it's worse than all others
        if epoch > patience and val_losses[-1] > max(val_losses[-patience:]):
            if verbose:
                print("Validation loss has not improved in {} epochs, stopping training at epoch {}".format(
                    patience, epoch+1))
            
            if logfile_fp is not None:
                logfile_fp.write("Validation loss has not improved in {} epochs, stopping training at epoch {}\n".format(
                    patience, epoch+1))
            break
        
    if verbose:
        print("Training finished. Best validation loss of {} at epoch {}".format(min(val_losses), best_epoch_num+1))
    if logfile_fp is not None:
        logfile_fp.write("Training finished. Best validation loss of {} at epoch {}\n".format(min(val_losses), best_epoch_num+1))

    # Save the final model if path given
    if model_save_folder is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_name = timestamp + "final_epoch_{}".format(best_epoch_num+1)
        torch.save(best_model.state_dict(), os.path.join(model_save_folder, save_name))
        if verbose:
            print("Saving final model weights from epoch {}".format(best_epoch_num+1))
        
        if logfile_fp is not None:
            logfile_fp.write("Saving final model weights from epoch {}\n".format(best_epoch_num+1))

            # Close the logfile
            logfile_fp.close()

    return best_model, best_epoch_num, train_losses, val_losses


def evaluate_model(model, test_dataset, batch_size=32, verbose=False, logfolder=None):
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Set the model to evaluation mode
    model.eval()

    # Make lists to store the losses
    losses = []

    # Loop over the batches
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):

            # Move the data to device
            X = X.to(model.device).float()
            y = y.to(model.device).float()

            # Forward pass
            output = model(X)

            # Calculate the loss
            loss = compute_loss(output, y, model.num_classes)

            losses.append(loss.item())
    
    # Average the losses
    avg_loss = sum(losses) / len(losses)

    if logfolder is not None:
        # Make logfolder if it doesn't exist
        if not os.path.exists(logfolder):
            os.makedirs(logfolder)

        # Make a new logfile with the current timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        logfile = os.path.join(logfolder, timestamp + "_evaluation.txt")
        logfile_fp = open(logfile, 'w')
    else:
        logfile_fp = None

    if verbose:
        print("Average loss: {}".format(avg_loss))
    
    if logfile_fp is not None:
        logfile_fp.write("Average loss: {}\n".format(avg_loss))
        
        # Close the logfile
        logfile_fp.close()
    
    return avg_loss
