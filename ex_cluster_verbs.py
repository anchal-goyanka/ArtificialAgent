from nltk import cluster
from nltk.cluster import euclidean_distance
from numpy import array
import csv
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
#from nltk.corpus import genesis
#genesis_ic = wn.ic(genesis, False, 0.0)

vectors = [] #list of list of path simirity of a verb with all other verbs
less_verbs = ['calling', 'dialing', 'searching', 'browsing', 'planning', 'setting', 'going', 'coming', 'locating', 'finding', 'send', 'emailing', 'messaging']
for verb in less_verbs:
    temp_vector = []
    a = WordNetLemmatizer().lemmatize(verb,'v') + '.v' + '.01'
    print a, wn.synset(a).definition()
    for word in less_verbs:
       b = WordNetLemmatizer().lemmatize(word,'v') + '.v' + '.01'
       temp_vector.append(wn.synset(a).jcn_similarity(wn.synset(b), semcor_ic))
    vectors.append(temp_vector)
K = 3
vec = [array(f) for f in vectors]
clusterer = cluster.KMeansClusterer(K, euclidean_distance, avoid_empty_clusters=True)
clusterer.cluster(vec, True)
ans_list = []
for i in range(K):
    ans_list.append([])
    
# classify a new vector
#print(clusterer.classify(vec[2]))
for i in range(len(vec)):
    ans_list[clusterer.classify(vec[i])].append(less_verbs[i])

for i in range(len(ans_list)):
    print ans_list[i]
