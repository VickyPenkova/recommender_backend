import time

from efficientnet.tfkeras import EfficientNetB7
from tensorflow.keras.models import load_model
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.optimizers import Adam, SGD

# parameters for optimizers
import Constants

lr = 1e-3


class EfficientNetB7Model:
    efficientnet_model = None

    def __init__(self, model_file_path=None):
        if model_file_path is None:
            self.efficientnet_model = EfficientNetB7(weights='imagenet')
            # ,include_top=False,pooling='max'
            self.efficientnet_model.trainable = False
            self.save_model('./model.h5')
        else:
            self.efficientnet_model = load_model(model_file_path)

        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
        self.efficientnet_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    def __get_embeddings(self, img_path):
        img = load_img(img_path, target_size=(300, 300))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        return self.efficientnet_model.predict(img).reshape(-1)

    def get_model_summary(self):
        return self.efficientnet_model.summary()

    def apply_embeddings(self, df):
        """
        Get Embeddings for all items in dataset

        NOTE: The operation takes more that an hour to complete
        :return: df_embs
        """
        df = df.reset_index(drop=True)
        start_time = time.time()
        # Make image_path feature
        df['image_path'] = df['id'].apply(lambda row: Constants.DATA_PATH + str(row) + '.jpg')
        # TODO: Split the images into batches and use multi-processing ??
        # map_embeddings = df['image_path'].apply(lambda img_path: self.__get_embeddings(img_path))
        map_embeddings = pd.read_csv('/Users/vpenkova/Documents/University/Diploma/map_embeddings.csv')
        # df_embs.head()
        #df_embs = map_embeddings.apply(pd.Series)
        df_embs = pd.read_csv('/Users/vpenkova/Documents/University/Diploma/embeddings.csv')
        print("--- %s seconds ---" % (time.time() - start_time))
        print(df_embs.shape)
        return df_embs

    def save_model(self, file_path, as_json=False):
        self.efficientnet_model.save(file_path)
        self.efficientnet_model.summary()
        if as_json:
            model_json = self.efficientnet_model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)