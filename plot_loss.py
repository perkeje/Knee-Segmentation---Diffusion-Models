import json
import matplotlib.pyplot as plt

# Read the log_loss.json file
with open("./results/checkpoints/loss_log.json", "r") as f:
    data = json.load(f)

# Extract epochs and train_loss values
epochs = [entry["epoch"] for entry in data]
train_loss = [entry["train_loss"] for entry in data]

# Plot the loss figure
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Training Loss", color="b", marker="o")

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig("train_loss.png")

# Show the plot
plt.show()
