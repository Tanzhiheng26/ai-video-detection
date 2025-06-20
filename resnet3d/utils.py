import imageio
import numpy as np
from IPython.display import Image

def to_gif(images, gif_name="sample.gif"):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave(gif_name, converted_images, fps=10)
  return Image(gif_name)