from utils import constants, dataLoad
from torch.utils.data import DataLoader
from models import poseNet, modelUtils

def load_data_test(verbose=True):
    if verbose: 
        print("Loading data...")
    dataset = dataLoad.load_data(verbose=verbose)

    # Get the first 2 images, and first 2 labels and print them out
    # Use dataloader with batch size 2
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    first_batch = next(iter(data_loader))

    if verbose:
        print("First batch of images: {}".format(first_batch[0]))
        print("First batch of labels: {}".format(first_batch[1]))

    return dataset

def forward_pass_tests(data, verbose=True):
    # Make a model 
    feature_dim = 3
    dropout_rate = 0.2
    # Determine the device
    device = constants.get_device()
    model = poseNet.PoseNet(feature_dim, dropout_rate, device=device)

    # Make a dataloader for a single batch of size 2
    data_loader = DataLoader(data, batch_size=2, shuffle=True, num_workers=0)
    first_batch = next(iter(data_loader))

    # Pass the first batch through the model
    output = model(first_batch[0].to(device))

    if verbose:
        print("Output: {}".format(output))

    # Test the output shape
    assert output.shape == (2, 6)

    # Test the loss function
    loss = modelUtils.regression_loss(output, first_batch[1].to(device))

    if verbose:
        print("Loss: {}".format(loss))
    
    # Test the loss shape
    assert loss.shape == (1,)
    
def run_all_tests():
    print("Running data load tests ... \n")
    data = load_data_test(verbose=True)
    print("Done with data load tests!")

    print("\n\n --- \n\n")
    print("Running model tests ... \n")
    forward_pass_tests()
    print("Done with model tests!")
