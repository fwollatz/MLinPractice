#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:43:57 2021

@author: ml
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_FOLLOWER_COUNT
import numpy as np
import tweepy



  


# class for extracting the amount of hashtags as a feature
class FollowerCount(FeatureExtractor):
    
    
    # assign the values accordingly
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""
    
    
    def __init__(self, input_column: str):
        """
        

        Parameters
        ----------
        input_column : str
            the column which includes the username

        Returns
        -------
        None.

        """

        
        super().__init__([input_column], COLUMN_FOLLOWER_COUNT)
    
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
        # import the module
        
          

          
        # authorization of consumer key and consumer secret
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
          
        # set access to user's access key and access secret 
        auth.set_access_token(self.access_token, self.access_token_secret)
          
        # calling the api 
        api = tweepy.API(auth)
        
        


        list_of_followers=[]
        for i in range(0,len(inputs[0])):
            
            # the screen name of the user
            screen_name = inputs[0][i]
              
            # fetching the user
            user = api.get_user(screen_name)
              
            # fetching the followers_count
            followers_count = user.followers_count
            list_of_followers+=[followers_count]

        result = np.array(list_of_followers)
        
        result = result.reshape(-1,1)
        return result
