import sys
import matplotlib.pyplot as plt


def plot_graph(data, filename):
    train_losses, val_losses, accuracies = data
    x_axis = range(len(train_losses))
    plt.plot(x_axis, train_losses, label="Train Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.plot(x_axis, accuracies, label="Accuracy")
    plt.legend()
    plt.savefig(f'{filename}.png')

def parse(filename):
    train_losses = []
    accuracies = []
    val_losses = []

    with open(f'{filename}.out', 'r') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            train_loss = float(lines[i].split()[3])
            train_losses.append(train_loss)
            val_loss = float(lines[i].split()[5])
            val_losses.append(val_loss)
            accuracy = float(lines[i].split()[-1])
            accuracies.append(accuracy)


    return train_losses, val_losses, accuracies

if __name__ == "__main__":
    filename = sys.argv[1]
    data = parse(filename)
    plot_graph(data, filename)