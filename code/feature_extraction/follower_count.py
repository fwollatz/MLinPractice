#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:43:57 2021

@author: ml
"""


from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_FOLLOWER_COUNT
import numpy as np
import time
import tweepy
import pickle


  


# class for extracting the amount of hashtags as a feature
class FollowerCount(FeatureExtractor):
    
    
    # assign the access-codes for the twitter api accordingly
    consumer_key = "VeoDsQOrGzPmTE1p6dLRklt5L"
    consumer_secret = "TRvLoRp7sK9aVVxQKvGhgY2zb282jBQhP7CFxICPlacbOVB9vI"
    access_token = "2800148739-vPpKSDACS4fJHnuOnkFjBo7CUDaehSBS3DzaFf9"
    access_token_secret="PJ3zVkeHxZz8AyiQweQ7u5AD3X5oVituu9Cpt1s8gLdR6"
    
    try:
        # authorization of consumer key and consumer secret
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
          
        # set access to user's access key and access secret 
        auth.set_access_token(access_token, access_token_secret)
          
        # calling the api 
        api = tweepy.API(auth)
    except:
        print("something went wrong with the connection to the tweepy api. Have you filled in your credentials correctly in code/feature_extraction/follower_count")
    
    #create or open a dict, to reduce stress on api and internet connection
    try:
        a_file = open("id_to_follower.pkl", "rb")
        id_to_follower = pickle.load(a_file)
        a_file.close()
    except:
        print("The file could not be loaded. Try to recreate the id_to_follower dict. This IS GOING TO take forever.")
        id_to_follower={}
    
    
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
        a_file = open("id_to_follower.pkl", "wb")
        pickle.dump(self.id_to_follower, a_file)
        a_file.close()
        print("Saved the followercounts.")
        
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
            array with unix_times of the individual tweets

        """
    

        list_of_follower_counts=[]
        
        #counter for persons whose follower data could not be accessed
        missing_data=0
        
        
        
        
        #get the follower count of every user
        for i in range(0,len(inputs[0])):
            # the screen name of the user
            user_id = int(inputs[0][i])
            
            #check if user is allready in dict, otherwise ask api
            if user_id in self.id_to_follower.keys():                  
                list_of_follower_counts+=[self.id_to_follower[user_id]]
            else:
                too_many_requests=True
                while too_many_requests:
                    too_many_requests=False
                    try:     
                        # fetching the user
                        user = self.api.get_user(user_id=user_id)
                          
                        # fetching the followers_count
                        followers_count = user.followers_count
                        list_of_follower_counts+=[followers_count]
                        self.id_to_follower[user_id]=followers_count
                    #User could not be found
                    except tweepy.errors.NotFound:
                         missing_data+=1
                         list_of_follower_counts+=[0]
                         self.id_to_follower[user_id]=0
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
                        self.id_to_follower[user_id]=0
                        print("there has been an error with this user (id={0}). His value is set to 0 followers.".format(user_id))
            #save the followercounts in external file
            if i%100==0:
                print(i, end="")
                self.safe_id_to_follower_to_pickle()

                
        print("there have been {0} people whose profiles we could not farm.".format(missing_data))
        
        #save the followercounts in external file
        self.safe_id_to_follower_to_pickle()
        
        result = np.array(list_of_follower_counts)
        
        result = result.reshape(-1,1)
        return result
