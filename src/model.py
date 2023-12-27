# isort was ran
import ssl

from keras import Input, layers, models
from keras.applications import EfficientNetB0, MobileNetV3Large, MobileNetV3Small
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, ReLU
from keras.models import Sequential

# needed to create this to validate my ssl context to load
# pre-trained models
ssl._create_default_https_context = ssl._create_unverified_context

NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)
OUT_CHANNELS = 64


# CNN i have created for this task
def myCNN():
    model = Sequential(
        [  # specifying dimensions of input
            Input(shape=INPUT_SHAPE),
            # 2D convulution filter with 64 6x6 kernels and a filter shift, or stride, of 2x2 ran over images.
            # rectified linear unit activation function applied to solve the vanishing gradient
            # issue (makes the gradient that is passed to earlier functions in backpropagation non-vanishing)
            Conv2D(
                OUT_CHANNELS,
                kernel_size=3,
                strides=2,
                padding="same",
                data_format="channels_last",
                activation="relu",
            ),
            # adapted method from GoogLeNet's Inception V1 code, this reduces the
            # spatial dimensions of the data while retaining the important features using pooling,
            # very strong with image data. found at:
            # https://ai.plainenglish.io/googlenet-inceptionv1-with-tensorflow-9e7f3a161e87
            layers.MaxPooling2D(3, strides=2),
            # ensures that the activation functions are normalized for the layer
            BatchNormalization(),
            # process repeats for layer 2 of CNN
            Conv2D(
                OUT_CHANNELS,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(3, strides=2),
            BatchNormalization(),
            # data is flattened from a 3D vector to 1D, prepares it for dense layer
            Flatten(),
            # produces the final classification output, softmax activation is
            # applied to convert the models output to probabilities for each class
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # compiling model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def cnn():
    model = Sequential(
        [  # specifying dimensions of input
            Input(shape=INPUT_SHAPE),
            Conv2D(
                64,
                kernel_size=3,
                strides=2,
                padding="same",
                data_format="channels_last",
                activation="relu",
            ),
            Conv2D(
                64,
                kernel_size=3,
                strides=2,
                padding="same",
                data_format="channels_last",
                activation="relu",
            ),
            layers.MaxPooling2D(padding="same"),
            layers.BatchNormalization(),
            Conv2D(
                128,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
            ),
            Conv2D(
                128,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(padding="same"),
            layers.BatchNormalization(),
            Flatten(),
            Dense(2048),
            Dense(2048),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# a simple CNN, one 2D convolution filter, one ReLU activation,
# noramlize, flatten and output.
def cnn2():
    model = Sequential(
        [
            Conv2D(
                1,
                kernel_size=3,
                input_shape=INPUT_SHAPE,
                activation="relu",
                data_format="channels_last",
                padding="same",
                use_bias=True,
            ),
            ReLU(),
            BatchNormalization(),
            Flatten(),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# importing EfficientNet model with pre-trained weights from imagenet
effnet = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=INPUT_SHAPE,
)


# added a few layers to effnet that will help with the current task
def effnetModel():
    model = effnet.output
    # global average pooling takes the average of the feature maps across spatial dimensions
    model = layers.GlobalAveragePooling2D()(model)
    # dropout helps with minimizing overfitting
    model = layers.Dropout(rate=0.5)(model)
    # added dense layer so outputs are correct for my task
    model = layers.Dense(4, activation="softmax")(model)
    model = models.Model(inputs=effnet.input, outputs=model)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# importing MobileNetV3(small) model with pre-trained weights from imagenet
mobile_netSmall = MobileNetV3Small(
    INPUT_SHAPE,
    weights="imagenet",
    include_top=False,
)
# freezes the weights of the pre-trained layers
# mobile_netSmall.trainable = False


def mobile_netV3Small():
    model = mobile_netSmall
    # global average pooling takes the average of the feature maps across spatial dimensions
    model = layers.GlobalAveragePooling2D()(mobile_netSmall.output)
    # dropout helps with minimizing overfitting
    model = layers.Dropout(rate=0.5)(model)
    # added dense layer so outputs are correct for my task
    model = layers.Dense(4, activation="softmax")(model)
    model = models.Model(inputs=mobile_netSmall.input, outputs=model)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# importing MobileNetV3(large) model with pre-trained weights from imagenet
mobile_netLarge = MobileNetV3Large(
    INPUT_SHAPE,
    include_top=False,
    weights="imagenet",
)


def mobile_netV3Large():
    model = mobile_netLarge
    # global average pooling takes the average of the feature maps across spatial dimensions
    model = layers.GlobalAveragePooling2D()(mobile_netLarge.output)
    # dropout helps with minimizing overfitting
    model = layers.Dropout(rate=0.5)(model)
    # added dense layer so outputs are correct for my task
    model = layers.Dense(4, activation="softmax")(model)
    model = models.Model(inputs=mobile_netLarge.input, outputs=model)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model
