#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that computes the hour of the day, seperated in 8 3-hour-slots
Created on Friday Oct 8 2021

@author: fwollatz
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_HOUR, SUFFIX_HOURS
import numpy as np



  


# class for extracting the amount of hashtags as a feature
class Hour(FeatureExtractor):
    
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


        super().__init__([input_column], COLUMN_HOUR)
    
    # don't need to fit, so don't overwrite _set_variables()
    
    def get_feature_name(self)->list:
        """
        

        Returns
        -------
        list
            list of all featurenames

        """
        suffixes=SUFFIX_HOURS
        names=[]
        for i in range(0,8):
            names+=[COLUMN_HOUR+"_"+suffixes[i]]
        return names
    
    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the hour of the day based on the inputs
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with hour of the individual tweets

        """

        
        list_of_hours=[]

        #splitting up the input
        times=inputs[0]
        

        for i in range(0,len(times)):

            #create hour from the string
            split_time=times[i].split(":")
            hour=int(split_time[0])



            #CREATE THE ONE HOT ENCODING (OHE)
            #           0      1     2     3    4      5     6     7     
            #           0-2    3-5  6-8  9-11 12-14 15-17 18-20 21-23
            ohe_hour=[False,False,False,False,False,False,False,False]
        
            ohe_hour[hour//3]=True
            
            
            #add to list
            list_of_hours+=[ohe_hour]
            
            

        result = np.array(list_of_hours)
        

        return result
