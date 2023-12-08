import subprocess

def testHyperparams():

    # Hyperparameter values we will test
    epochs = [20, 50]
    patiences = [5, 10]
    batch_sizes = [32, 64, 256]
    learning_rates = [0.001, .005]

    # Testing loop
    for e in epochs:
        for p in patiences:
            for bs in batch_sizes:
                for lr in learning_rates:
                    subprocess.run(f'python trainModel.py -e {e} -p {p} -b {bs} -lr {lr}')

if __name__ == '__main__':
    testHyperparams()