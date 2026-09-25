# hiragana net, in plain C

**This README.md was written by an AI agent because I was too lazy to describe what this thing does. Why not to leave it without README.md? Dunno, repo seems more complete with it. Don't pay attention to history of 2-3 commits, I had wrote this thing like months ago and just didn't push this to github somewhy.**

I wanted to see if I could write a neural net without PyTorch doing the thinking for me. This is the result: a small network that looks at a 128x128 grayscale image and tells you which of the 46 hiragana it is.

Matrix math goes through Apple's Accelerate, so it only builds on a Mac. Python is only there to turn images into bytes. All the training happens in `matrixes.c`.

## files

**`engine.c`** came first. Two weights, a loss I wrote by hand, gradient descent with momentum. No images, no layers. I just wanted to watch the numbers move before I had to deal with backprop.

**`matrixes.c`** is the actual network: 16384 → 512 → 128 → 64 → 46, ReLU on the hidden layers, softmax at the end. It trains on one random sample per step with cross-entropy and a learning rate of 0.001, then saves the weights to `hiragana_model.bin`. If that file doesn't exist yet, it starts from random weights.

The Python helpers:
- `prepare_dataset.py` reads `dataset/<class>/*.jpg`, resizes everything to 128x128 and writes `dataset.bin`
- `convert.py` does the same for a single `my_drawing.jpg` and writes `custom_image.bin`
- `plot.py` turns `training_loss.txt` into `learning_curve.png`

`run` is just `matrixes.c` compiled.

## building

```
clang -O2 -framework Accelerate matrixes.c -o run
clang engine.c -o engine
```

For the Python side you need pillow, numpy and matplotlib.

## running it

Put your images in folders named after the character (`dataset/aa`, `dataset/chi`, ...). The folder name is the label. Then:

```
python prepare_dataset.py
./run
```

You get a small menu:

1. **train**: asks for the number of epochs, logs the loss every 100 steps, saves the model
2. **guess a random training image**
3. **guess your own drawing**: run `python convert.py` first to make `custom_image.bin`
4. **quit**

After training, `python plot.py` draws the loss curve.
