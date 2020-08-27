import tensorflow as tf

# Augmentation Function
def flip(x):
    """
    flip image
    """
    x = tf.image.random_flip_left_right(x)
    return x

def color(x):
    """
    Color change
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x - tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x):
    """
    Rotation image
    """
    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))
    return x

def zoom(x, scale_min=1.2, scale_max=1.4):
    """
    Zoom image
    """
    h, w, c = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)
    sh = h * scale
    sw = w * scale
    x = tf.image.resize(x, (sh, sw))
    x = tf.image.resize_with_crop_or_pad(x, h, w)
    return x

def standardization(x):
    return tf.image.per_image_standardization(x)

# Data Preprocessing Function

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(record):
    feature_dict = tf.io.parse_single_example(record, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image'])
#     image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.
    
    label = feature_dict['label']
    label = tf.one_hot(label, 3)
    
    return image, label

def _parse_augmentation_function(record):
    feature_dict = tf.io.parse_single_example(record, feature_description)
    image = tf.io.decode_png(feature_dict['image'])
#     image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.
    
    # Image Augmentation function
    image = flip(image)
#     image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(image), lambda: image)
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(image), lambda: image)
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(image), lambda: image)
    
    label = feature_dict['label']
    label = tf.one_hot(label, 3)
    
    return image, label

def _parse_standard_function(record):
    feature_dict = tf.io.parse_single_example(record, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image'])
#     image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.image.per_image_standardization(image)
    
    label = feature_dict['label']
    label = tf.one_hot(label, 3)
    
    return image, label

def _parse_augmentation_standard_function(record):
    feature_dict = tf.io.parse_single_example(record, feature_description)
    image = tf.io.decode_png(feature_dict['image'])
#     image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.image.per_image_standardization(image)
    
    # Image Augmentation function
    image = flip(image)
#     image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: color(image), lambda: image)
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: rotate(image), lambda: image)
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: zoom(image), lambda: image)
    
    label = feature_dict['label']
    label = tf.one_hot(label, 3)
    
    return image, label