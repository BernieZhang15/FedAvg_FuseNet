import numpy as np


def generateLocalEpochs(percentage, size, max_epochs):
    if percentage == 0:
        return np.array([max_epochs] * size)
    else:
        # get the number of clients to have fewer than E epochs
        heterogenous_size = int((percentage / 100) * size)

        # generate random uniform epochs of heterogenous size between 1 and E
        epoch_list = np.random.randint(1, max_epochs, heterogenous_size)

        # the rest of the clients will have E epochs
        remaining_size = size - heterogenous_size
        rem_list = [max_epochs] * remaining_size

        epoch_list = np.append(epoch_list, rem_list, axis=0)

        # shuffle the list and return
        np.random.shuffle(epoch_list)

        return epoch_list
