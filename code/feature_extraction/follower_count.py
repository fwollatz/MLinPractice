#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:43:57 2021

@author: ml
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_FOLLOWER_COUNT
import numpy as np
import pickle
import time
import tweepy



  


# class for extracting the amount of hashtags as a feature
class FollowerCount(FeatureExtractor):
    
    
    # assign the access-codes for the twitter api accordingly
    _consumer_key = input("please write your consumer key for the twitter api here:")
    _consumer_secret = input("please write your _consumer_secret for the twitter api here:")
    _access_token = input("please write your _access_token for the twitter api here:")
    _access_token_secret=input("please write your _access_token_secret for the twitter api here:")
    
    try:
        # authorization of consumer key and consumer secret
        _auth = tweepy.OAuthHandler(_consumer_key, _consumer_secret)
          
        # set access to user's access key and access secret 
        _auth.set_access_token(_access_token, _access_token_secret)
          
        # calling the api 
        _api = tweepy.API(_auth)
    except:
        print("something went wrong with the connection to the tweepy api. Have you filled in your credentials correctly in code/feature_extraction/follower_count")
    
    #create or open a dict, to reduce stress on api and internet connection
    try:
        a_file = open("id_to_follower.pkl", "rb")
        _id_to_follower = pickle.load(a_file)
        a_file.close()
    except:
        print("The file could not be loaded. Try to recreate the id_to_follower dict. This IS GOING TO take forever.")
        _id_to_follower={}
    
    
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
    

    def safe_id_to_follower_to_pickle(self):
        """
        safes the _id_to_follower dictionary to a pickle

        Returns
        -------
        None.

        """
        a_file = open("id_to_follower.pkl", "wb")
        pickle.dump(self._id_to_follower, a_file)
        a_file.close()

        
    def _get_values(self, inputs: list) -> np.ndarray :
        """
        compute the followercount of the user
        
        Parameters
        ----------
        inputs : list
            list of all tweets

        Returns
        -------
        result : np.ndarray
            array with followercounts

        """
    

        list_of_follower_counts=[]
        
        #counter for persons whose follower data could not be accessed
        missing_data=0
        
        
        
        
        #get the follower count of every user
        for i in range(0,len(inputs[0])):
            # the screen name of the user
            user_id = int(inputs[0][i])
            
            #check if user is allready in dict, otherwise ask api
            if user_id in self._id_to_follower.keys():                  
                list_of_follower_counts+=[self._id_to_follower[user_id]]
            else:
                too_many_requests=True
                while too_many_requests:
                    too_many_requests=False
                    try:     
                        # fetching the user
                        user = self._api.get_user(user_id=user_id)
                          
                        # fetching the followers_count
                        followers_count = user.followers_count
                        list_of_follower_counts+=[followers_count]
                        self._id_to_follower[user_id]=followers_count
                    #User could not be found
                    except tweepy.errors.NotFound:
                         missing_data+=1
                         list_of_follower_counts+=[0]
                         self._id_to_follower[user_id]=0
                         print("The User {0} could not be found. His value is set to 0 followers.".format(user_id))
                    #Twitter has been asked to many things at once.
                    except tweepy.errors.TooManyRequests :
                        too_many_requests=True
                        print("The rate of requests has been to high. We will try again in 60 seconds. To speed things up: download the pickle id_to_follower.pkl from our git") 
                        self.safe_id_to_follower_to_pickle()
                        time.sleep(60)
                    except:
                        missing_data+=1
                        list_of_follower_counts+=[0]
                        self._id_to_follower[user_id]=0
                        print("there has been an error with this user (id={0}). His value is set to 0 followers.".format(user_id))
            #save the followercounts in external file
            if i%100==0:
                self.safe_id_to_follower_to_pickle()

                
        print("there have been {0} people whose profiles we could not farm.".format(missing_data))
        
        #save the followercounts in external file
        self.safe_id_to_follower_to_pickle()
        
        result = np.array(list_of_follower_counts)
        
        result = result.reshape(-1,1)
        return result
