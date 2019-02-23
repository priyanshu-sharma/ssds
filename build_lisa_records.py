##This is build_lisa_records.py which will be used to accept an input set of images, creating the training and
##testing splits, and then create TensorFlow compatible record files that can be used for training.

from config import lisa_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os

def main(_):

    f = open(config.CLASSES_FILE, "w")
    for (k, v) in config.CLASSES.items():
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: ’" + k + "’\n"
                "}\n")

        f.write(item)

    f.close()

    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    for row in rows[1:]:
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        if label not in config.CLASSES:
            continue

        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)
    datasets = [("train", trainKeys, config.TRAIN_RECORD), ("test", testKeys, config.TEST_RECORD)]

    for (dType, keys, outputPath) in datasets:
        print("[INFO] processing ’{}’...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0

        for k in keys:
            encoded = tf.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            for (label, (startX, startY, endX, endY)) in D[k]:
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                """Don't know
                image = cv2.imread(k)
                startX = int(xMin * w)
                startY = int(yMin * h)
                endX = int(xMax * w)
                endY = int(yMax * h)

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.imshow("Image", image)
                cv2.waitKey(0)

                """

                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                total += 1

            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

        writer.close()
        print("[INFO] {} examples saved for ’{}’".format(total, dType))

if __name__ == "__main__":
    tf.app.run()
