import numpy as np
import re
import tensorflow as tf
from nltk import cluster
from nltk.cluster import euclidean_distance
from numpy import array
import csv
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.wsd import lesk

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

for i in range(len(data_list)):
    for j in range(len(data_list[i][0])):
        data_list[i][0][j] = apps_id_dict[data_list[i][0][j]]
#print data_list ex. [[[21,3,4], 'go to gallery'], ...]

per = []
loc = []
verbs = []
nouns= []
#task_verb_dict = {}
verbs_with_sense = []
verb_tasks_dict = {}
#a lsit of list of tokens used to represent a task description
task2tokens = []
#extracting nouns and verbs from each task description
for i in data_list:
    temp =[]
    description = i[1]
    tokens = nltk.word_tokenize(description)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    for j in range(len(entities)):
        if type(entities[j]) == nltk.Tree:
            if entities[j].label() in ['PERSON']:
                per.append((entities.leaves())[j])
                temp.append((entities.leaves())[j])
            elif entities[j].label() in ['LOCATION']:
                loc.append((entities.leaves())[j])
                temp.append((entities.leaves())[j])
        else:
            if entities[j][1][0] == 'V':
                #finding most suitable word sense
                syn = (lesk(description.split(), entities[j][0] , 'v'))
                if syn is not None:
                    name = syn.name()
                    verbs_with_sense.append(str(name))
                    temp.append(str(name))
                    if str(name) not in verb_tasks_dict.keys():
                        verb_tasks_dict[str(name)] = []
                        verb_tasks_dict[str(name)].append(description)
                    else:
                        verb_tasks_dict[str(name)].append(description)
                verbs.append(entities[j][0])
            elif entities[j][1][0] == 'N':
                nouns.append(entities[j][0])
                temp.append(entities[j][0])
    task2tokens.append(temp)
set_nouns = set(nouns)
set_verbs = set(verbs)
#print len(set_verbs), 'len of set of original verbs'#526
nouns_once = list(set_nouns)
verbs_once = list(set_verbs)
verbs_with_sense = list(set(verbs_with_sense))
#print len(verbs_with_sense), 'len of new verbs with sense'#469

#list of all nouns present in wordnet database
all_lemma_word = []
for synset in list(wn.all_synsets('n')):
    word = synset.name().split('.')[0]
    all_lemma_word.append(word)
'''
#list of all verbs present in wordnet database
all_lemma_verbs = []
for synset in list(wn.all_synsets('v')):
    word = synset.name().split('.')[0]
    all_lemma_verbs.append(word)
'''
diff_nouns = []
other_nouns = []
for i in nouns_once:
    normalized_noun = WordNetLemmatizer().lemmatize(i,'n')
    s = normalized_noun + '.' + 'n' + '.' + '01'
    loc_ = 'location.n.01'
    per_ = 'person.n.01'
    if normalized_noun not in all_lemma_word:
        other_nouns.append(i)
        continue
    location_hypernym =  wn.synset(loc_).lowest_common_hypernyms(wn.synset(s))
    person_hypernym =  wn.synset(per_).lowest_common_hypernyms(wn.synset(s))
    if location_hypernym[0] == wn.synset(loc_):
        loc.append(i)
    elif person_hypernym[0] == wn.synset(per_):
        per.append(i)
    else:
        diff_nouns.append(i)
        
#creating a dictionary with keys as nouns and values as their frequency in data
noun_freq_dict = {}
for noun in nouns:
    if noun not in noun_freq_dict.keys():
        noun_freq_dict[noun] = 1
    else:
        noun_freq_dict[noun]+=1
# 901 print len(noun_freq_dict.keys())
# 901 print len(per) + len(loc) + len(other_nouns) + len(diff_nouns)
#sorts keys in a dict based on values
#To do: Find a statistical number instead of 20 based on the nouns vocabulary
most_freq_nouns = sorted(noun_freq_dict, key=noun_freq_dict.get)[-21:-1]
final_nouns = []
for i in most_freq_nouns:
    if i in loc:
        continue
    elif i in per:
        continue
    else:
        final_nouns.append(i)
most_freq_nouns = final_nouns

#tfile = open('nouns.txt', 'w')
#print>>tfile, 'MOST FREQUENT NOUNS-->'
#print>>tfile, most_freq_nouns
#print>>tfile, 'PERSON-->'
#print>>tfile, per
#print>>tfile, 'LOCATION-->'
#print>>tfile, loc
#print>>tfile, 'OTHER NOUNS(NOT USED)-->'
#print>>tfile, other_nouns
#print>>tfile, 'NOUN FREQUENCY DICT-->'
#print>>tfile, noun_freq_dict

vectors = [] #list of list of path simirity of a verb with all other verbs
for verb in verbs_with_sense:
    temp_vector = []
    for word in verbs_with_sense:
        temp_vector.append(wn.synset(verb).path_similarity(wn.synset(word)))
    vectors.append(temp_vector)
K = 25
vec = [array(f) for f in vectors]
clusterer = cluster.KMeansClusterer(K, euclidean_distance, avoid_empty_clusters=True)
clusterer.cluster(vec, True)
ans_list = []
for i in range(K):
    ans_list.append([])
    
# classify a new vector
for i in range(len(vec)):
    ans_list[clusterer.classify(vec[i])].append([verbs_with_sense[i], verb_tasks_dict[verbs_with_sense[i]]])

'''
for i in range(len(ans_list)):
    print 'BEGINNIG OF A NEW CLUSTER'
    print ' '
    for j in range(len(ans_list[i])):
        print 'beginnign of a new verb sense'
        print ''
        print ans_list[i][j]

tfile = open('verb_sense.txt1', 'w')
for item in ans_list:
  print>>tfile, ' ' 
  print>>tfile, 'BEGINNING OF A NEW VERB SENSE'
  temp = []
  for j in item:
    print>>tfile, '  '
    print>>tfile,  'nextttttttttttttttttt'
    #print>>tfile, j
    temp.append(j[0])
  print>>tfile, temp 
'''

#indexing each word within outer list
verb_index_dict = {}
for i in range(len(ans_list)):
    for pair in ans_list[i]:
        verb_index_dict[pair[0]] = i

input_size = K + 2 + len(final_nouns) # per, loc, last_20 - per - loc
normalized_tasks = []

for j in range(len(data_list)):
    normalized_tasks.append([[0 for i in range(input_size)], data_list[j][1], []])

for i in range(len(data_list)):
    tokens = task2tokens[i]
    for j in tokens:
        if j in per:
            normalized_tasks[i][2].append(j)
            normalized_tasks[i][0][K] += 1
        elif j in loc:
            normalized_tasks[i][2].append(j)
            normalized_tasks[i][0][K+1] += 1
        elif j in final_nouns:
            normalized_tasks[i][2].append(j)
            normalized_tasks[i][0][final_nouns.index(j) + K + 2] += 1
        elif j in verb_index_dict.keys():
            normalized_tasks[i][2].append(j)
            normalized_tasks[i][0][verb_index_dict[j]] += 1

#tfile = open('normalized_tasks4.txt', 'w')
#for item in normalized_tasks:
  #print>>tfile, ' '
  #print>>tfile, item

K_ = 20
input_array = []
output_array = []   
for i in range(len(data_list)):
   input_array.append(array(normalized_tasks[i][0])) 
   output_array.append([0 for i in range(K_)])  
clusterer = cluster.KMeansClusterer(K_, euclidean_distance, avoid_empty_clusters=True)
clusterer.cluster(input_array, True)
task_clusters = []
for i in range(K_):
    task_clusters.append([])
for i in range(len(input_array)):
    task_clusters[clusterer.classify(input_array[i])].append([data_list[i][1]])
    output_array[i][clusterer.classify(input_array[i])] = 1
output_array = np.array(output_array, dtype=np.float32)
input_array = np.array(input_array)
#tfile = open('task_clusters4.txt', 'w')
#for item in task_clusters:
  #print>>tfile, item
  #print>>tfile, ' '

num_hidden_nodes = 10

graph = tf.Graph()
with graph.as_default():

  tf_train_dataset = tf.constant(input_array[:900, :],dtype=np.float32)
  tf_train_labels = tf.constant(output_array[:900], dtype=np.float32)
  tf_test_dataset = tf.constant(input_array[900:, :], dtype=np.float32)
  
  
  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([input_size, num_hidden_nodes]))
  biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  weights2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes, K_]))
  biases2 = tf.Variable(tf.zeros([K_]))
  
  # Training computation.
  lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits = tf.matmul(lay1_train, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  #lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  #valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
  lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

num_steps = 100
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):

    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 50 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, output_array[:900, :]))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), output_array[900:, :]))
