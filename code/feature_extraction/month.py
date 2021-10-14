#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that computes the month of the tweets release in the given column.

Created on Friday Oct 8 2021

@author: fwollatz
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_MONTH, SUFFIX_MONTHS
import numpy as np



  


# class for extracting the amount of hashtags as a feature
class Month(FeatureExtractor):
    
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


        super().__init__([input_column], COLUMN_MONTH)
    
    # don't need to fit, so don't overwrite _set_variables()
    
    def get_feature_name(self):
        suffixes=SUFFIX_MONTHS
        names=[]
        for i in range(0,12):
            names+=[COLUMN_MONTH+"_"+suffixes[i]]
        return names
    
    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the unixtime based on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with month of the individual tweets

        """

        
        list_of_months=[]

        #splitting up the input
        dates=inputs[0]
        

        for i in range(0,len(dates)):

            #create month from the string
            splitdate=dates[i].split("-")
            month=int(splitdate[1])



            #CREATE THE ONE HOT ENCODING (OHE)
            #           0      1     2     3    4      5     6     7     8     9     10    11
            #           JA    FEB   MA    APR   MAI   JUN   JUL   AUG   SEP   OKT   NOV   DEZ
            ohe_month=[False,False,False,False,False,False,False,False,False,False,False,False]
        
            ohe_month[month-1]=True
            
            
            #add to list
            list_of_months+=[ohe_month]
            
            

        result = np.array(list_of_months)
        

        return result
