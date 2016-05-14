# -*- coding: utf-8 -*-
import re
import numpy
#import read_dataset as rd

#users_tweets = rd.get_tweets()

def time_decsec(h,m,s):
    time_sec_dec = int(h) * 3600 + int(m) * 60 + int(s)
    return time_sec_dec

def time_dechour(h,m,s):
    time_hour_dec = int(h) + int(m) / 60 + int(s) / 3600
    return time_hour_dec

def week_un_num(n):
    if 0 < int(n) <= 7: 
        wun = 1
    if 7 < int(n) <= 14: 
        wun = 2
    if 14 < int(n) <= 21: 
        wun = 3
    else:
        wun = 4
    return wun

def time_YMDHM(y,w,d,h,minute):
    time_total = int(y) * 524160 + week_un_num(w) *10080 + int(d) * 1440 + int(h) * 60 + int(minute)
    return time_total

def month_num(m):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mn = months.index(m) + 1
    return mn

def AM_or_PM(n):
    if 0 <= int(n) < 12: 
        aop = 'AM'
    else:
        aop = 'PM'
    return aop

def YMD_num(n):
    nymd = []
    for i in n:
        loop = int(i)
        nymd.append(loop)
        

def week_unique(n):
    if 0 < int(n) <= 7: 
        wu = "A"
    if 7 < int(n) <= 14: 
        wu = "B"
    if 14 < int(n) <= 21: 
        wu = "C"
    else:
        wu = "D"
    return wu

def season(n):
    spring = ['Mar', 'Apr', 'May']
    summer = ['Jun', 'Jul', 'Aug']
    fall = ['Sep', 'Oct', 'Nov']
    winter = ['Dec','Jan','Feb']
    if n in spring:
        ses = 'spring'
    if n in summer:
        ses = 'summer'
    if n in fall:
        ses = 'fall'
    if n in winter:
        ses = 'winter'
    return ses


def hour_sit_tweets(sec,twe):   #groups tweets and times tweeted, for those output within a 60 min span, input ymdhm (min) time
    sits = [[sec[0]]]
    pits = [[twe[0]]]
    for x,y in (zip(sec[1:],twe[1:])):
        if abs(x - sits[-1][-1]) <= 60:
            sits[-1].append(x)
            pits[-1].append(y)

        else:
            sits.append([x])
            pits.append([y])

    return sits,pits  


def long_tweet_time(chuck):      #takes in tuple output from hour_sit_tweets 
    s = []
    p = []
    for x,y in zip(chuck[0],chuck[1]):
        for i,j in zip((range(len(x)-1)),y):
            s.append((x[i] - x[i+1]))
            p.append(len(j))
    return s,p

def composing_long_tweets(ch): #avg minutes it takes to compose a long tweet (>120 characters)
    clt = []
    for i,j in zip(ch[0],ch[1]):
        if j > 120:
            clt.append(i)
    if len(clt) > 0:
        comp_time = (sum(clt) / len(clt))
    else:
        comp_time = 0
    return comp_time

def span_ten(sec): #input ymdhm time (minutes)
    sits = [[sec[0]]]
    for x in sec[1:]:
        if abs(x - sits[-1][-1]) <= 10:
            sits[-1].append(x)
        else:
            sits.append([x])
    return sits  

def span_thirty(sec):
    sits = [[sec[0]]]
    for x in sec[1:]:
        if abs(x - sits[-1][-1]) <= 30:
            sits[-1].append(x)
        else:
            sits.append([x])
    return sits

def span_hour(sec):
    sits = [[sec[0]]]
    for x in sec[1:]:
        if abs(x - sits[-1][-1]) <= 60:
            sits[-1].append(x)
        else:
            sits.append([x])
    return sits

def spancount(span):
    count_span = [len(i) for i in span if len(i) > 1]
    if len(count_span) > 0:
        comp_time = (sum(count_span) / len(count_span))
    else:
        comp_time = 0
    return comp_time 
    
def timecount(tim):
    count_tim = [(max(i) - min(i)) for i in tim if len(i) > 1]
    if len(count_tim) > 0:
        comp_time = (sum(count_tim) / len(count_tim))
    else:
        comp_time = 0
    return comp_time

def tweet_time(users_tweets):
    times = {}
    times["control"] = {}
    times["schizophrenia"] = {}
    
    match_all = re.compile(r'(^\w{3})\s(\w{3})\s(\d{2})\s(\d{2})\:(\d{2})\:(\d{2})\s(\+\d{4})\s(\d{4}$)')
    
    for key in users_tweets.keys():
        for k in users_tweets[key].keys():
            times[key][k] = {}

            ymd_str = [(match_all.match(timestamp).group(8) + str(month_num(match_all.match(timestamp).group(2))) + \
            match_all.match(timestamp).group(3)) for timestamp in users_tweets[key][k]["timestamps"]] #unique day-month-year identifier
            
            my = [(match_all.match(timestamp).group(2) + match_all.match(timestamp).group(8)) \
            for timestamp in users_tweets[key][k]["timestamps"]] #unique month-year identifier
            
            wmy = [(week_unique(match_all.match(timestamp).group(3)) + match_all.match(timestamp).group(2) + \
            match_all.match(timestamp).group(8)) for timestamp in users_tweets[key][k]["timestamps"]]
 
            tdh = [time_dechour(match_all.match(timestamp).group(4), match_all.match(timestamp).group(5), \
            match_all.match(timestamp).group(6)) for timestamp in users_tweets[key][k]["timestamps"]]
            
            ymdhm = [time_YMDHM(match_all.match(timestamp).group(8),\
            match_all.match(timestamp).group(3),\
            match_all.match(timestamp).group(3),\
            match_all.match(timestamp).group(4),\
            match_all.match(timestamp).group(5)) for timestamp in users_tweets[key][k]["timestamps"]]
  
            ap = [AM_or_PM(match_all.match(timestamp).group(4)) for timestamp in users_tweets[key][k]["timestamps"]]
            seas = [season(match_all.match(timestamp).group(2)) for timestamp in users_tweets[key][k]["timestamps"]]
        
            times[key][k]["avg_posting_time"] = numpy.mean(tdh) #avg time of day when you tweet
            times[key][k]["frac_AM_posts"] = ap.count('AM') / len(ap)
            times[key][k]["frac_winter_posts"] = seas.count('winter') / len(seas)
            times[key][k]["frac_summer_posts"] = seas.count('summer') / len(seas)
            
            times[key][k]["daily_tweeting_rate"] = len(users_tweets[key][k]["tweets"]) / len(set(ymd_str))
            times[key][k]["weekly_tweeting_rate"] = len(users_tweets[key][k]["tweets"]) / len(set(wmy))
            times[key][k]["monthly_tweeting_rate"] = len(users_tweets[key][k]["tweets"]) / len(set(my))
            
            times[key][k]["10_min_span_tweets"] = spancount(span_ten(ymdhm)) #avg number of tweets sent in span of 10-min frequency (tweets sent within 10 min of preceding tweet)
            times[key][k]["10_min_span_time"] = timecount(span_ten(ymdhm)) #avg time span in minutes, of a sitting
            
            times[key][k]["30_min_span_tweets"] = spancount(span_thirty(ymdhm)) 
            times[key][k]["30_min_span_time"] = timecount(span_ten(ymdhm))            

            times[key][k]["60_min_span_tweets"] = spancount(span_hour(ymdhm)) 
            times[key][k]["60_min_span_time"] = timecount(span_ten(ymdhm))
            
            times[key][k]["comp_120ch_twt_60sit"] = composing_long_tweets(long_tweet_time\
            (hour_sit_tweets(ymdhm,users_tweets[key][k]["tweets"])))
            
    return times


    




            
            
    
    

        
    
    
