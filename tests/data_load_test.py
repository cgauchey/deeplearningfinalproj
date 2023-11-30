from utils import constants, data_load
from torch.utils.data import DataLoader

def load_data_test(verbose=True):
    if verbose: 
        print("Loading data...")
    dataset = data_load.load_data(verbose=verbose)

    # Get the first 2 images, and first 2 labels and print them out
    # Use dataloader with batch size 2
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    first_batch = next(iter(data_loader))

    if verbose:
        print("First batch of images: {}".format(first_batch[0]))
        print("First batch of labels: {}".format(first_batch[1]))

    return dataset

def run_all_tests():
    print("Running data load tests ... \n")
    load_data_test(verbose=True)
    print("Done with data load tests!")
