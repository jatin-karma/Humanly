import tensorflow as tf
from tensorflow.keras import layers

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["human", "non_human"]


def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_image")

    
    x = layers.RandomFlip("horizontal", name="augment_flip")(inputs)
   
    x = layers.RandomZoom((-0.2, 0.2), name="augment_zoom")(x)
    
    x = layers.RandomRotation(0.05, name="augment_rotation")(x)

    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_scale")(x)

    #Pretrained Base Network
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False

    #Classifier Head 
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="non_human_probability")(x)

    return tf.keras.Model(inputs, outputs, name="human_face_classifier")