from visdom import Visdom
import numpy as np
import math
import os.path

viz = Visdom()

textwindow = viz.text("Hello Pytorch")
image_window = viz.image(
    np.random.rand(3,256,256),
    opts=dict(
        title = "random",
        caption = "random noise"
    )
)
images_window = viz.images(
    np.random.rand(10,3,64,64),
    opts=dict(
        title = "random",
        caption = "random noise"
    )
)