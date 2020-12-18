# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from dataloaders.Dataloader import Dataloader
from get_recommender import get_recommender
from util import Util

from model.EfficientNetB7 import EfficientNetB7Model

MODEL_FILE_PATH = './model/model.h5'

df = Dataloader.load_csv('./data/styles.csv')


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def recommend_similar_items(item_id, top_n=5):
    efficientnet_model = EfficientNetB7Model(MODEL_FILE_PATH)
    #
    df_embs = efficientnet_model.apply_embeddings(df)
    return get_recommender(item_id, df, df_embs, top_n)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Recommend similar items
    idx_ref = 200
    # Recommendations
    idx_rec, idx_sim = recommend_similar_items(idx_ref, top_n=6)
    Util.show_image(df.iloc[idx_ref].image_path)
    # print_hi('PyCharm')
