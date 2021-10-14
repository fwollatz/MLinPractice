#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that computes the day of the week of the tweet in the given column.

Created on Friday Oct 8 2021

@author: fwollatz
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_WEEKDAY, SUFFIX_WEEKDAY
import datetime
import numpy as np


  


# class for extracting the day of the week as a feature
class Weekday(FeatureExtractor):
    
    def __init__(self, input_column: str):
        """
        

        Parameters
        ----------
        input_column : str
            the column which includes a date in this format:  2020-03-05.

        Returns
        -------
        None.

        """


        super().__init__([input_column], COLUMN_WEEKDAY)
    
    # don't need to fit, so don't overwrite _set_variables()
    
    def get_feature_name(self):
        suffixes=SUFFIX_WEEKDAY
        names=[]
        for i in range(0,7):
            names+=[COLUMN_WEEKDAY+"_"+suffixes[i]]
        return names
    
    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the day of the week on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with weekdays of the individual tweets

        """

        
        list_of_weekdays=[]

        #splitting up the input
        dates=inputs[0]
        

        for i in range(0,len(dates)):

            #create datetime from the string
            splitdate=dates[i].split("-")
            year=int(splitdate[0])
            month=int(splitdate[1])
            day=int(splitdate[2])
    
            date_time = datetime.datetime(year, month, day)
            
            #create day of the week from date
            day_nr=date_time.weekday()


            #CREATE THE ONE HOT ENCODING (OHE)
            #           0     1     2     3     4    5     6
            #           MO    DI    MI    DO    FR   SA    SO
            ohe_week=[False,False,False,False,False,False,False]
            
            ohe_week[day_nr]=True

            
            #add to list
            list_of_weekdays+=[ohe_week]
            
            

        result = np.array(list_of_weekdays)
        
        #result = result.reshape(-1,1)
        return result
