#no of individual apps i s 188 and k is 20
#wil be useful if 18 reduced to 40 

from nltk import cluster
from nltk.cluster import euclidean_distance
from numpy import array
import csv
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

with open('R2_reformat.csv', 'rU') as infile:
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
      for header, value in row.items():
        try:
          data[header].append(value)
        except KeyError:
          data[header] = [value]

#list of list containing task description and corresponding app names
data_list = []
for i in range(len(data['no'])):
    data_list.append([data['app_names'][i].split(';'), data['task_description'][i]])

#Generating a dictionary with keys as appnames and values as new app ids
apps_id_list = []
for i in data_list:
    for app_name in i[0]:
        if app_name not in apps_id_list:
            apps_id_list.append(app_name)
#print len(apps_id_list) #188
apps_id_dict = {}#dict with keys as appnames and values as new app ids
for i in range(len(apps_id_list)):
    apps_id_dict[apps_id_list[i]] = i+1 #app ids starting from 1

#storing all app names in a file
#thefile = open('app_names.txt', 'w')
#for item in apps_id_dict.keys():
  #print>>thefile, item


for i in range(len(data_list)):
    for j in range(len(data_list[i][0])):
        data_list[i][0][j] = apps_id_dict[data_list[i][0][j]]
#print data_list ex. [[[21,3,4], 'go to gallery'], ...]

size = len(apps_id_dict.keys())
print size
array_list = []
for i in range(len(data_list)):
    temp = [0 for i in range(size)]
    for j in data_list[i][0]:
        temp[j] += 1
    temp = array(temp)
    array_list.append(temp)
print type(array_list[9])
K_ = 30
clusterer = cluster.KMeansClusterer(K_, euclidean_distance, avoid_empty_clusters=True)
clusterer.cluster(array_list, True)
task_clusters = []
for i in range(K_):
    task_clusters.append([])
for i in range(len(array_list)):
    task_clusters[clusterer.classify(array_list[i])].append([data_list[i][1]])
tfile = open('cluster_using_apps20.txt', 'w')
for item in task_clusters:
  print>>tfile, item
  print>>tfile, ' '

