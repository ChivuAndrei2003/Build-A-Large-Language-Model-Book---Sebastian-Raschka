import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, axl = plt.subplots(figsize=(5, 3))
    axl.plot(epochs_seen, train_losses, label="Training loss")
    axl.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    axl.set_xlabel("Epochs")
    axl.set_ylabel("Loss")
    axl.legend(loc="upper right")
    axl.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = axl.twiny()  # Creates a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
