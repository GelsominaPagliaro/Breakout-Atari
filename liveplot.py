import matplotlib.pyplot as plt
import os
from datetime import datetime


class LivePlot:

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch x 10")
        self.ax.set_ylabel("Returns")
        self.ax.set_title("Return over Epochs")
        self.data = None
        self.epochs = 0
        self.eps_data = None

    def update_plot(self, stats):
        self.data = stats["AvgReturns"]
        self.eps_data = stats["EpsilonCheckPoint"]

        self.epochs = len(self.data)
        self.ax.clear()
        self.ax.set_xlim(0, self.epochs)

        self.ax.plot(self.data, "b-")
        self.ax.plot(self.eps_data, "r-")
        self.ax.legend(loc='upper left')

        if not os.path.exists("plots"):
            os.makedirs("plots")

        current_date = datetime.now().strftime("%Y-%m-%d")

        self.fig.savefig(f"plots/plot_{current_date}.png")

