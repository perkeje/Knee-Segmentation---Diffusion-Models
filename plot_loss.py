import json
import matplotlib.pyplot as plt

# Read the log_loss.json file
with open("./results/checkpoints/log.json", "r") as f:
    data = json.load(f)
# Extract steps, train_loss, val_loss, val_f1, and val_iou values
steps = [entry["step"] + 1 for entry in data]  # Adding 1 to each step
train_loss = [entry["loss"] for entry in data]
val_loss = [entry["val_loss"] for entry in data]
val_f1 = [entry["val_f1"] for entry in data]
val_iou = [entry["val_iou"] for entry in data]

# Plot the loss figure on a logarithmic scale
plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, label="Trening Gubitak", color="b")
plt.plot(steps, val_loss, label="Val Gubitak", color="r")

plt.xlabel("Epoha")
plt.ylabel("Gubitak (log skala)")
plt.title("Trening i Validacijski Gubitak Kroz Epohe (Logaritamski Prikaz)")
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.grid(True)

# Save the loss plot to a file
plt.savefig("train_val_loss_log.png")

# Show the loss plot
plt.show()

# Plot the loss figure
plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, label="Trening Gubitak", color="b")
plt.plot(steps, val_loss, label="Val Gubitak", color="r")

plt.xlabel("Epoha")
plt.ylabel("Gubitak")
plt.title("Trening i Validacijski Gubitak Kroz Epohe")
plt.legend()
plt.grid(True)

# Save the loss plot to a file
plt.savefig("train_val_loss.png")

# Show the loss plot
plt.show()

# Plot the F1 and IoU figure
plt.figure(figsize=(12, 6))
plt.plot(steps[20:], val_f1[20:], label="Val F1", color="g")
plt.plot(steps[20:], val_iou[20:], label="Val IoU", color="m")

plt.xlabel("Epoha")
plt.ylabel("Vrijednost Metrike")
plt.title("Val F1 i IoU Kroz Epohe")
plt.legend()
plt.grid(True)

# Save the F1 and IoU plot to a file
plt.savefig("val_f1_iou.png")

# Show the F1 and IoU plot
plt.show()
