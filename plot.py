import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("training_loss.txt")
epochs = data[:, 0]
loss = data[:, 1]
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label="Cross-Entropy Loss", color="blue")
plt.title("Hiragana detection training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("learning_curve.png")
print("[+] Plot saved as learning_curve.png")
plt.show()
