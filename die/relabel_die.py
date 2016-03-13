from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
'''I  did not label everything correct in my automated generation script.

Most notably some 5s should be 1s or 3s, due to lighting issues.
This script generated die.y.improved.npy.
'''

X = np.load('die.X.npy')
y = np.load('die.y.npy')

y_improved_fname = 'die.y.improved.npy'

# Need to correct the labels --- they were not entirely correct.
# Do to this, display the image, bind space to keep same, and number to renumber.
# Store the improved labels
improved_y = []
i = 0

def plot(i):
    x = X[i]
    ax.set_title(y[i])
    ax.imshow(np.rollaxis(x, 0, 3))
    fig.canvas.draw()

def press(event):
    print('Pressed key: "{}"'.format(event.key))
    global i
    if event.key == ' ':
        imp_y = y[i]
    else:
        imp_y = int(event.key)
    improved_y.append(imp_y)
    i += 1
    if i == len(X):
        np.save(y_improved_fname, np.array(improved_y))
    plot(i)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)
plot(i)

plt.show()
