import csv
import os

import pandas
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import Constants

STYLES_CSV_ROW__id = 'id'
STYLES_CSV_ROW__gender = 'gender'
STYLES_CSV_ROW__masterCategory = 'masterCategory'
STYLES_CSV_ROW__subCategory = 'subCategory'
STYLES_CSV_ROW__articleType = 'articleType'
STYLES_CSV_ROW__baseColour = 'baseColour'
STYLES_CSV_ROW__season = 'season'
STYLES_CSV_ROW__year = 'year'
STYLES_CSV_ROW__usage = 'usage'
STYLES_CSV_ROW__productDisplayName = 'productDisplayName'

IMAGES_CSV_ROW__filename = 'filename'
IMAGES_CSV_ROW__link = 'link'


class Dataloader:
    @staticmethod
    def load_csv(csv_file_pth, error_bad_lines=False):
        return pd.read_csv(csv_file_pth, error_bad_lines=error_bad_lines)

    @staticmethod
    def create_styles_csv(data_dict=None):
        """
        Creates styles.csv template if it doesn't exist from the styles.csv template.
        Populates the csv file with data
        :param data_dict: python dict with keys corresponding to the header rows in the csv template
        :return: None
        """
        # TODO: test
        # read_csv parameters explained https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        if os.path.exists(Constants.STYLES_CSV_PRODUCTION_PATH) and os.path.isfile(Constants.STYLES_CSV_PRODUCTION_PATH):
            df = pandas.read_csv(Constants.STYLES_CSV_PRODUCTION_PATH,
                                 index_col=None)
        else:
            df = pandas.read_csv(Constants.STYLES_CSV_TEMPLATE_PATH,
                                 index_col=None)

        if data_dict:
            df = df.append(data_dict, ignore_index=True)

        df.to_csv(Constants.STYLES_CSV_PRODUCTION_PATH)
