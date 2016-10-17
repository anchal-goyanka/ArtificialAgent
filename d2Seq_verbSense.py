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

#storing all app names in a file
#thefile = open('app_names.txt', 'w')
#for item in apps_id_dict.keys():
  #print>>thefile, item


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
#extracting nouns and verbs from each task description
for i in data_list:
    temp =[]
    description = i[1]
    tokens = nltk.word_tokenize(description)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    #print entities
    for j in range(len(entities)):
        if type(entities[j]) == nltk.Tree:
            #print entities, 'entities main', entities[j], 'is a subtree'
            if entities[j].label() in ['PERSON']:
                per.append((entities.leaves())[j])
            elif entities[j].label() in ['LOCATION']:
                loc.append((entities.leaves())[j])
        else:
            if entities[j][1][0] == 'V':
                #finding most suitable word sense
                syn = (lesk(description.split(), entities[j][0] , 'v'))
                if syn is not None:
                    name = syn.name()
                    verbs_with_sense.append(str(name))
                    if str(name) not in verb_tasks_dict.keys():
                        verb_tasks_dict[str(name)] = []
                        verb_tasks_dict[str(name)].append(description)
                    else:
                        verb_tasks_dict[str(name)].append(description)
                verbs.append(entities[j][0])
            elif entities[j][1][0] == 'N':
                nouns.append(entities[j][0])
 
set_nouns = set(nouns)
set_verbs = set(verbs)
print len(set_verbs), 'len of set of original verbs'#526
nouns_once = list(set_nouns)
verbs_once = list(set_verbs)
verbs_with_sense = list(set(verbs_with_sense))
print len(verbs_with_sense), 'len of new verbs with sense'#469
print verbs_with_sense[0]
print verbs_with_sense[3]

'''
new_list = []
for i in range(len(verbs_with_sense)):
    verbs_with_sense_str.append(str(verbs_with_sense[i]))
for i in verbs_with_sense_str:
    if i in new_list:
        continue
    else:
        new_list.append(i)
#print len(new_list)
#print new_list
'''
'''
#list of all nouns present in wordnet database
all_lemma_word = []
for synset in list(wn.all_synsets('n')):
    word = synset.name().split('.')[0]
    all_lemma_word.append(word)

#list of all verbs present in wordnet database
all_lemma_verbs = []
for synset in list(wn.all_synsets('v')):
    word = synset.name().split('.')[0]
    all_lemma_verbs.append(word)
'''
'''
other_nouns = []
for i in nouns_once:
    s = i + '.' + 'n' + '.' + '01'
    loc_ = 'location.n.01'
    per_ = 'person.n.01'
    if i not in all_lemma_word:
        other_nouns.append(i)
        continue
    #print 's in lemma word'
    location_hypernym =  wn.synset(loc_).lowest_common_hypernyms(wn.synset(s))
    person_hypernym =  wn.synset(per_).lowest_common_hypernyms(wn.synset(s))
    if location_hypernym[0] == wn.synset(loc_):
        loc.append(i)
    elif person_hypernym[0] == wn.synset(per_):
        per.append(i)

#creating a dictionary with keys as nouns and values as their frequency in data
noun_freq_dict = {}
for noun in nouns:
    if noun not in noun_freq_dict.keys():
        noun_freq_dict[noun] = 1
    else:
        noun_freq_dict[noun]+=1
#print noun_freq_dict
#sorts keys in a dict based on values
most_freq_nouns = sorted(noun_freq_dict, key=noun_freq_dict.get)[-20:-1]
#print most_freq_nouns, '20'
final_nouns = []
for i in most_freq_nouns:
    if i in loc:
        continue
    elif i in per:
        continue
    else:
        final_nouns.append(i)
#print per, loc, final_nouns
#tfile = open('nouns.txt', 'w')
#print>>tfile, '20 MOST FREQUENT NOUNS-->'
#print>>tfile, final_nouns
#print>>tfile, 'PERSON-->'
#print>>tfile, per
#print>>tfile, 'LOCATION-->'
#print>>tfile, loc
'''
'''
#extracting all words from set of appnames in the data, ex: games.mm --> [games, mm]
apps_word_list =[]
for i in apps_id_list:
    for j in i.split('.'):
        if j not in apps_word_list:
            apps_word_list.append(j)
#print apps_word_list
common_list = []
print len(nouns), 'lenof nouns'
for i in nouns:
    if i in apps_word_list:
        common_list.append(i)
final_noun_list = []
for i in common_list:
    if i in last_20:
        final_noun_list.append(i)


less_verbs = []
verbs_not_in_wordnet = []
#keeping only those verbs present in wordnet database
for i in verbs_once:
    #bringing verb in its base form
    lemma_word = WordNetLemmatizer().lemmatize(i,'v')
    if lemma_word in all_lemma_verbs:
        less_verbs.append(i)
    else:
        verbs_not_in_wordnet.append(i)
'''
vectors = [] #list of list of path simirity of a verb with all other verbs
for verb in verbs_with_sense:
    temp_vector = []
    #a = WordNetLemmatizer().lemmatize(verb,'v') + '.v' + '.01'
    for word in verbs_with_sense:
        #b = WordNetLemmatizer().lemmatize(word,'v') + '.v' + '.01'
        temp_vector.append(wn.synset(verb).path_similarity(wn.synset(word)))
    vectors.append(temp_vector)
K = 25
print len(vectors), 'len of vecs'
vec = [array(f) for f in vectors]
clusterer = cluster.KMeansClusterer(K, euclidean_distance, avoid_empty_clusters=True)
clusterer.cluster(vec, True)
ans_list = []
for i in range(K):
    ans_list.append([])
    
# classify a new vector
#print(clusterer.classify(vec[2]))
for i in range(len(vec)):
    ans_list[clusterer.classify(vec[i])].append([verbs_with_sense[i], verb_tasks_dict[verbs_with_sense[i]]])


for i in range(len(ans_list)):
    print 'BEGINNIG OF A NEW CLUSTER'
    print ' '
    for j in range(len(ans_list[i])):
        print 'beginnign of a new verb sense'
        print ''
        print ans_list[i][j]

 
tfile = open('verb_clusters_25_no_empty.txt', 'w')
for item in ans_list:
  print>>tfile, item
'''
#indexing each word within outer list
verb_index_dict = {}
for i in range(len(ans_list)):
    for verb in ans_list[i]:
        verb_index_dict[verb] = i

input_size = K + 2 + len(final_nouns) # per, loc, last_20
normalized_tasks = []

for j in range(len(data_list)):
    normalized_tasks.append([[0 for i in range(input_size)], data_list[j][1],[]])

for i in range(len(data_list)):
    tokens = nltk.word_tokenize(normalized_tasks[i][1])
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

tfile = open('normalized_tasks2.txt', 'w')
for item in normalized_tasks:
  print>>tfile, item
K_ = 20
array_list = []
for i in range(len(data_list)):
    array_list.append(array(normalized_tasks[i][0]))
clusterer = cluster.KMeansClusterer(K_, euclidean_distance, avoid_empty_clusters=True)
clusterer.cluster(array_list, True)
task_clusters = []
for i in range(K_):
    task_clusters.append([])
for i in range(len(array_list)):
    task_clusters[clusterer.classify(array_list[i])].append([data_list[i][1]])
tfile = open('task_clusters2.txt', 'w')
for item in task_clusters:
  print>>tfile, item
  print>>tfile, ' '
'''
