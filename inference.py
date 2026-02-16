import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_CHANNELS = 3

# encoder
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


# decoder
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# generator ==> encoder + decoder
def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# load input image
def load_image(image_file):
  """Load a single image file for inference."""
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32)
  return image
    
# generate output image
def generate_images(model, test_input, tar=None):
  prediction = model(test_input, training=True)
  
  if tar is not None:
    plt.figure(figsize=(15, 5))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    cols = 3
  else:
    plt.figure(figsize=(10, 5))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    cols = 2

  for i in range(cols):
    plt.subplot(1, cols, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

  return prediction[0]


# main inference function
def inference_generation(image_path, checkpoint_dir):
  # 1. Initialize Generator
  generator = Generator()

  # 2. Setup Checkpoint and Restore
  checkpoint = tf.train.Checkpoint(generator=generator)
  
  # Identify latest checkpoint
  latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_ckpt:
      print(f"Restoring from {latest_ckpt}...")
      checkpoint.restore(latest_ckpt).expect_partial()
  else:
      print(f"No checkpoint found in {checkpoint_dir}. Please check the path.")
      return

  # 3. Load and Preprocess Image
  if not os.path.exists(image_path):
      print(f"Error: Image file {image_path} not found.")
      return
  
  total_image = load_image(image_path)

  # Split side-by-side image
  w = tf.shape(total_image)[1]
  w_half = w // 2
  target_image = total_image[:, :w_half, :]
  input_image = total_image[:, w_half:, :]

  # Resize to 256x256 and normalize
  input_image = tf.image.resize(input_image, [256, 256])
  target_image = tf.image.resize(target_image, [256, 256])
  
  # Normalize to [-1, 1]
  input_image = (input_image / 127.5) - 1
  target_image = (target_image / 127.5) - 1
  
  # Add batch dimension
  input_image = tf.expand_dims(input_image, 0)
  target_image = tf.expand_dims(target_image, 0)

  # 4. Generate and Show Image
  print("Generating prediction...")
  prediction = generate_images(generator, input_image, target_image)
  return input_image[0], target_image[0], prediction


def main():
  # Define your paths here
  image_path = 'dataset/facades/test/20.jpg'
  checkpoint_dir = './models'

  inference_generation(image_path, checkpoint_dir)
  

#if __name__ == "__main__":
#    main()