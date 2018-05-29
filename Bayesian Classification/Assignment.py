# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:20:48 2017

@author: nvlas
"""

import os
import math
import string
from nltk.stem.snowball import SnowballStemmer


unique_words_set = set()

def stemming(unique_words_set):
    
    stemmer = SnowballStemmer("english")   
    stemmer.stem(unique_words_set)

def remove_punctuation(unique_words_set):
    
    unique_words_set = ''.join(c for c in unique_words_set if c not in string.punctuation)
    

def formula(prob,review):
    total_prob = 0
    for word in review:
        if word in prob:
            total_prob += math.log(prob[word])
    return total_prob 

def populate_probability(frequency_dictionary,unique_words_set):
    total_unique_words = len(unique_words_set)
    total_freqs = sum(frequency_dictionary.values())
    
    prob = {}
    for each_word in unique_words_set:
        if each_word in frequency_dictionary:
            prob[each_word] = frequency_dictionary.get(each_word) + 1 / (total_freqs + total_unique_words)
    return prob  


def load_test(positive_prob,negative_prob,test_type):
    neg_test = 0
    pos_test = 0
    total_pos = 0
    total_neg = 0
    
    path = "smallTest\\pos\\"
    
    if test_type == "neg":
        path = "smallTest\\neg\\"
        
    listing = os.listdir(path)
    for each_file in listing:
        f = open(path + each_file, "r", encoding="utf8")
        fd = f.read().lower().split()
        
        for review in fd:
             if review is not None:
                pos_test = formula(positive_prob,fd)
                neg_test = formula(negative_prob,fd)
        if pos_test > neg_test:
            total_pos+=1
        else:
            total_neg+=1
    if test_type == "pos":            
        print("Result of postive: " + str((total_pos/1000)*100) + "%") 
    else:
        print("Result of negative: " + str((total_neg/1000)*100) + "%") 
        
def stop_words(clean_set):
    f = open("stop_words.txt")
    stop_words = f.read()
    for each_word in stop_words:
        if each_word in clean_set:
            clean_set.remove(each_word)
    

def main():

    #Getting the path of positive and negative files
    pos_path = 'LargeIMDB\\pos\\'
    neg_path = 'LargeIMDB\\neg\\'
    
    neg_listing = os.listdir(neg_path)
    pos_listing = os.listdir(pos_path)
    
    #Dictionaries whit words frequency
    negative_frequency = {}
    positive_frequency = {}
    
    #List of pos and neg words
    neg_words = []
    pos_words = []
    
    #Loading each file from the negative directory
    #Split them up into each words 
    for each_neg_file in neg_listing:
        print('Current file is: ' + each_neg_file)
        f = open(neg_path + each_neg_file, "r", encoding="utf8")
        neg_words = f.read()
        for word in neg_words.lower().split():
            if word is not None:
                #Add them to the unique set of words
                unique_words_set.add(word)
                #Calculate the frequency of each word
                negative_frequency[word] = negative_frequency.setdefault(word,0) + 1
            
    #Loading each file from the positive directory
    #Split them up into each words 
    for each_pos_file in pos_listing:
        print('Current file is: ' + each_pos_file)
        f = open(pos_path + each_pos_file, "r", encoding="utf8")
        pos_words = f.read()
        for word in pos_words.lower().split():
            if word is not None:
                #Add them to the unique set of words
                unique_words_set.add(word)
                #Calculate the frequency of each word
                positive_frequency[word] = positive_frequency.setdefault(word,0) + 1
            
    #Populte the probabilities
    positive_prob = populate_probability(positive_frequency,unique_words_set)
    negative_prob = populate_probability(negative_frequency,unique_words_set)
    
    #Remove puntuantion
    remove_punctuation(unique_words_set)
    
    
    #Removing the stop words from the unique data set
    stop_words(unique_words_set)
    
    
    #print(unique_words_set)
    #Getting the % of the test 
    test_type = "pos"    
    load_test(positive_prob,negative_prob,test_type)
    test_type = "neg"
    load_test(positive_prob,negative_prob,test_type)
    
    
    
main()