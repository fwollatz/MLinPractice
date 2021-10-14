#!/usr/bin/env python3
#-*- coding : utf-8 -*-
# @Author : RCheng
# @Time :  08.10.21 11:02

import unittest
import pandas as pd
import numpy as np
from feature_extraction.check_photos_existence import PhotoChecker

from code.util import COLUMN_PHOTO_EXISTENCE


class PhotoCheckerTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COL = "photos"
        self.OUTPUT_COL = "contain_photos"
        self.photo_checker = PhotoChecker(self.INPUT_COL)

    #def test_amount_of_urls_correct(self):
     #   output = self.URL_feature.fit_transform(self.df)
     #   EXPECTED_COUNT = 2

     #   self.assertEqual(output[0][0], EXPECTED_COUNT)

    def test_input_col(self):
        self.assertEqual(self.photo_checker._input_columns, [self.INPUT_COL])

    def test_feature_name(self):
        self.assertEqual(self.photo_checker.get_feature_name(), COLUMN_PHOTO_EXISTENCE)

    def test_photos_existence(self):
        #arrange
        photo_col_input = list(["['https://pbs.twimg.com/media/Ey4gMWDUYAAdKdY.jpg']", "['HelloKitty.jpg']", "[]"])
        photo_col_preprocesed = np.array([1,1,0])
        print(photo_col_preprocesed.shape)

        input_df = pd.DataFrame()
        input_df[self.INPUT_COL] = photo_col_input

        #act
        photos_checked = self.photo_checker.fit_transform(input_df).squeeze()
        print("photos_checked equals to ", photos_checked.shape)
        
        #assert
        self.assertEqual(photos_checked[0], photo_col_preprocesed[0],
                         'Expected 1 when https://pbs... is present'
                         )
        self.assertEqual(photos_checked[1], photo_col_preprocesed[1],
                         '1 is expected when .jpg format is present'
                         )
        self.assertEqual(photos_checked[2], photo_col_preprocesed[2],
                         '0 is expected when a tweet contains no pictures'
                         )

if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.PhotoCheckerTest]__")
    unittest.main()
