# isort was ran
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.model import cnn, cnn2, effnetModel, mobile_netV3Large, mobile_netV3Small, myCNN
from src.preprocess import load_data


def main():
    # initialize batch size and epochs
    BATCH_SIZE = 32
    EPOCHS = 12

    # load data and labels
    data, labels = load_data()

    # various models to be tested
    models = [
        myCNN(),
        cnn(),
        cnn2(),
        effnetModel(),
        mobile_netV3Small(),
        mobile_netV3Large(),
    ]

    # labels for models
    model_names = [
        "MyCNN",
        "DeepCNN",
        "SimpleCNN",
        "Effnet",
        "MobileNetV3Small",
        "MobileNetV3Large",
    ]

    # initialize dataframe to store values in
    scores = pd.DataFrame(
        index=["error", "macro_avg_prec", "macro_avg_rec", "macro_avg_f1"],
        columns=["MyCNN", "DeepCNN", "SimpleCNN", "Effnet", "MobileNetV3Small", "MobileNetV3Large"],
        data=0.0,
    )

    # learning rate callback variable,
    # ReduceLROnPlateau adjusts the learning rate when monitor stops improving.
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.2,
        patience=2,
        min_delta=0.001,
        mode="auto",
        verbose=1,
    )

    # shuffle samples
    data, labels = shuffle(data, labels, random_state=40)

    # generate train-test splits
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(data, labels, test_size=0.2, random_state=40)

    # loop through models, set checkpoint file for model,
    # train and test, save statistics to json
    for i, model in enumerate(models):
        cur_model = model_names[i]
        checkpoint = ModelCheckpoint(
            f"output/{cur_model}.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="auto",
            verbose=1,
        )

        # summary of models layers, other model information
        model.summary()

        # fitting model
        training = model.fit(
            X_train,
            y_train,
            BATCH_SIZE,
            EPOCHS,
            validation_split=0.1,
            callbacks=[checkpoint, reduce_lr],
        )

        # get accuracy, validation accuracy and loss for each epoch
        acc_epochs, val_acc_epochs, loss_epochs = (
            training.history["accuracy"],
            training.history["val_accuracy"],
            training.history["loss"],
        )

        # plotting accuracy and validation accuracy over epochs
        plt.plot(range(1, EPOCHS + 1), acc_epochs, marker="o", label="Training Accuracy")
        plt.plot(range(1, EPOCHS + 1), val_acc_epochs, marker="o", label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Over Epochs - {cur_model}")
        plt.legend()
        plt.savefig(f"figs/accuracy/{cur_model}_acc_epochs.png")
        plt.clf()

        # plotting loss over epochs
        plt.plot(range(1, EPOCHS + 1), loss_epochs, marker="o", label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Over Epochs - {cur_model}")
        plt.legend()

        # save figures
        plt.savefig(f"figs/loss/{cur_model}_loss_epochs.png")
        plt.clf()

        # evaluate model, predict aswell to get precision, recall and f1
        score = model.evaluate(X_test, y_test, BATCH_SIZE, verbose=True)
        pred = model.predict(X_test)

        # get labels out of encoded matricies
        pred = np.argmax(pred, axis=1)
        y_test_max = np.argmax(y_test, axis=1)

        # names of classes for plotting
        labels = [
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor",
        ]

        # heatmap of classes with predictions within each cell
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        sns.heatmap(
            confusion_matrix(y_test_max, pred),  # get predicted labels
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,  # put number of times model predicted that class within cells
            cmap="Blues",
            linewidths=2,
            linecolor="black",
        )
        # set text of figure
        fig.text(
            s=f"Heatmap of the Confusion Matrix - {cur_model}",
            size=18,
            fontweight="bold",
            fontname="monospace",
            color="black",
            y=0.9,
            x=0.2,
        )

        # save figures
        plt.savefig(f"figs/heatmaps/{cur_model}_class_hmap.png")
        plt.clf()

        # initializing dictionaries to store statistics about each class
        glioma = {}
        meningioma = {}
        no_tumor = {}
        pituitary = {}
        values_dict = {}

        # get classification report, which returns precision, recall, f1 score,
        # as well as the macro and weighted averages of these for each class
        report = classification_report(y_test_max, pred, output_dict=True)
        glioma["prec"] = round(report["0"]["precision"], 3)
        glioma["recall"] = round(report["0"]["recall"], 3)
        glioma["f1"] = round(report["0"]["f1-score"], 3)

        meningioma["prec"] = round(report["1"]["precision"], 3)
        meningioma["recall"] = round(report["1"]["recall"], 3)
        meningioma["f1"] = round(report["1"]["f1-score"], 3)

        no_tumor["prec"] = round(report["2"]["precision"], 3)
        no_tumor["recall"] = round(report["2"]["recall"], 3)
        no_tumor["f1"] = round(report["2"]["f1-score"], 3)

        pituitary["prec"] = round(report["3"]["precision"], 3)
        pituitary["recall"] = round(report["3"]["recall"], 3)
        pituitary["f1"] = round(report["3"]["f1-score"], 3)

        # add all class dictionaries to this one for ease of plotting
        values_dict["glioma"] = glioma
        values_dict["meningioma"] = meningioma
        values_dict["no_tumor"] = no_tumor
        values_dict["pituitary"] = pituitary

        xAxis = np.arange(len(values_dict) - 1)
        width = 0.2
        multiplier = 0

        fig, axis = plt.subplots(layout="constrained")

        # turning dict into list for easier access
        classes = list(values_dict.items())
        for clas, values in classes:
            offset = width * multiplier
            rects = axis.bar(xAxis + offset, list(values.values()), width, label=clas)
            axis.bar_label(rects, padding=3)
            multiplier += 1

        axis.set_ylabel("Percentage")
        axis.set_title(f"Precision, Recall and F1 for each class - {cur_model}")
        axis.set_xticks(xAxis + width, values_dict["glioma"].keys())
        axis.legend(loc="lower right", ncols=3)
        axis.set_ylim(0, 1.1)

        plt.tight_layout()  # so all tick labels are shown properly
        plt.savefig(f"figs/scores/{cur_model}_scores_epochs.png")
        # clear figure so no aesthetics get carried over to future plots
        plt.clf()

        # grab error rate for model
        err = round(1 - score[1], 4)
        # get macro averages from classification report
        macro_avg_prec = report["macro avg"]["precision"]
        macro_avg_rec = report["macro avg"]["recall"]
        macro_avg_f1 = report["macro avg"]["f1-score"]
        # add all values to dataframe
        scores.loc["error", cur_model] = err
        scores.loc["macro_avg_prec", cur_model] = macro_avg_prec
        scores.loc["macro_avg_rec", cur_model] = macro_avg_rec
        scores.loc["macro_avg_f1", cur_model] = macro_avg_f1

    scores.to_json("output/scores.json")


if __name__ == "__main__":
    main()
