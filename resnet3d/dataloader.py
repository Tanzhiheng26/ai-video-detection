import random
import cv2
import numpy as np
import tensorflow as tf

class FrameGenerator:
  def __init__(self, path, n_frames, training = False, seed = 42):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.seed = seed
    random.seed(self.seed)

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = 1 if name == "1_fake" else 0 # Encode labels
      yield video_frames, label


def format_frames(frame):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32) # scaled to [0,1)
  frame = tf.image.resize(frame, [256, 256])
  # Center crop to 224x224
  frame = tf.image.resize_with_crop_or_pad(frame, 224, 224)
  return frame

def frames_from_video_file(video_path, n_frames, frame_step = 4):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start) # endpoint is inclusive

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    # read the last frame for every frame_step frames
    if ret:
      frame = format_frames(frame)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  # reorder the color channels of video frames from BGR to RGB
  result = np.array(result)[..., [2, 1, 0]]

  return result
