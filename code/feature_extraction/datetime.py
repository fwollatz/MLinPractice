#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that computes the unixtime of the tweet in the given column.

Created on Thu Oct 7 

@author: fwollatz
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_DATETIME_UNIX
import datetime
import numpy as np
import time



  


# class for extracting the amount of hashtags as a feature
class DateTime(FeatureExtractor):
    
    def __init__(self, input_column_date: str, input_column_time: str):
        """
        

        Parameters
        ----------
        input_column_date : str
            the column which includes a date in this format:  2020-03-05.
        input_column_time : str
            the column which includes a time in this format: 09:20:18.

        Returns
        -------
        None.

        """


        super().__init__([input_column_date,input_column_time], COLUMN_DATETIME_UNIX)
    
    # don't need to fit, so don't overwrite _set_variables()
    

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
            array with unix_times of the individual tweets

        """

        
        list_of_unix_times=[]

        #splitting up the input
        dates=inputs[0]
        times=inputs[1]
        

        for i in range(0,len(dates)):

            #create datetime from the two strings
            splitdate=dates[i].split("-")
            year=int(splitdate[0])
            month=int(splitdate[1])
            day=int(splitdate[2])
            
            splittime=times[i].split(":")
            hour=int(splittime[0])
            minute=int(splittime[1])
            second=int(splittime[2])
            
            date_time = datetime.datetime(year, month, day, hour, minute, second)
            print(date_time)
            #transform datetime to unix-time            
            unix_date_time=time.mktime(date_time.timetuple())
            
            #append to array
            list_of_unix_times+=[unix_date_time]
            

        result = np.array(list_of_unix_times)
        
        result = result.reshape(-1,1)
        return result
