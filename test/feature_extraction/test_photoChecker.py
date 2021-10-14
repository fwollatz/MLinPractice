#!/usr/bin/env python3
#-*- coding : utf-8 -*-
# @Author : RCheng
# @Time :  08.10.21 11:02

import unittest
import pandas as pd
from feature_extraction.check_photos_existence import PhotoChecker


class PhotoCheckerTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COL = "photos"
        self.OUTPUT_COL = "contain_photos"
        self.photo_checker = PhotoChecker()

    def test_input_col(self):
        self.assertEqual(self.photo_checker._output_column, self.OUTPUT_COL)

    def test_photos_existence(self):
        #arrange
        photo_col_input = list(["['https://pbs.twimg.com/media/Ey4gMWDUYAAdKdY.jpg']", "['HelloKitty.jpg']", "[]"])
        photo_col_preprocesed = [1,1,0]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COL] = photo_col_input

        #act
        photos_checked = self.photo_checker.fit_transform(input_df)
        print("photos_checked equals to ", photos_checked)
        
        #assert
        self.assertEqual(photos_checked[self.OUTPUT_COL][0], photo_col_preprocesed[0],
                         'Expected 1 when https://pbs... is present'
                         )
        self.assertEqual(photos_checked[self.OUTPUT_COL][1], photo_col_preprocesed[1],
                         '1 is expected when .jpg format is present'
                         )
        self.assertEqual(photos_checked[self.OUTPUT_COL][2], photo_col_preprocesed[2],
                         '0 is expected when a tweet contains no pictures'
                         )

if __name__ == '__main__':
    print("__[RUNNING: test.preprocessing.PhotoCheckerTest]__")
    unittest.main()
