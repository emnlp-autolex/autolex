import pyconll
import utils
import numpy as np
np.random.seed(1)
from collections import defaultdict
import codecs
import os
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import json
import pandas as pd


def get_vocab_from_set(input):
    word_to_id = {}
    if "NA" in input:
        word_to_id['NA'] = 0
    for i in input:
        if i == "NA":
            continue
        word_to_id[i] = len(word_to_id)
    id_to_word = {v:k for k,v in word_to_id.items()}
    return word_to_id, id_to_word

class DataLoader(object):
    def __init__(self, args, relation_map):
        self.args = args
        self.feature_dictionary = {}
        self.model_dictionary = {}
        self.relation_map = relation_map
        self.feature_map = {}
        self.triple_freq = defaultdict(lambda: 0)
        self.model_data = defaultdict(lambda : 0)
        self.model_data_case = {}
        self.feature_map_model = {}
        self.feature_freq = {}
        self.model_feature_label = {}
        self.remove_features = ['Gender', 'Person', 'Number']
        self.required_relations = []
        self.wals_features = {'subject-verb': ['subj_VERB'],
                              'object-verb': ['obj_VERB'],
                              'noun-adposition':['NOUN_ADP', 'PRON_ADP', 'PROPN_ADP'], #When adp is the syntactic head
                              'adjective-noun': ['ADJ_mod_NOUN', 'ADJ_mod_PROPN', 'ADJ_mod_PRON'],
                              'numeral-noun': ['NUM_mod_NOUN', 'NUM_mod_PROPN', 'NUM_mod_PRON'],
                                }
        random.seed(args.seed)

    def unison_shuffled_copies(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def isValidFeature(self, pos, head_pos, relation):
        model_features = defaultdict(list)
        for model, features in self.wals_features.items():
            for feature_type in features:
                info = feature_type.split("_")
                if len(info) == 3: #dep-pos-relation-head-pos
                    if info[0] == pos and info[2] == head_pos and info[1] in relation:
                        model_features[model].append(feature_type)
                elif len(info) == 2: #dep-relation, dep-head
                    if info[0] in relation and info[1] == head_pos:
                        model_features[model].append(feature_type)
                    if info[0] == pos and info[1] == head_pos:
                        model_features[model].append(feature_type)

        return model_features

    def isValidLemma(self, lemma, upos):
        if upos in ['PUNCT', 'NUM', 'PROPN' 'X', 'SYM']:
            return None
        if lemma:
            lemma = lemma.lower()
            lemma = lemma.replace("\"", "").replace("\'", "")
            if lemma == "" or lemma == " ":
                return None
            else:
                return lemma
        return None

    def readData(self, inputFiles, genreparseddata, wikidata, vocab_file, spine_word_vectors, spine_features, spine_dim):
        #print("Parsing the input file!")
        for num, inputFile in enumerate(inputFiles):
            if inputFile is None:
                continue
            genreparseddata_ = genreparseddata[num]


            self.lang_full = inputFile.strip().split('/')[-2].split('-')[0][3:]
            f = inputFile.strip()
            data = pyconll.load_from_file(f"{f}")

            is_test = False
            if "test" in inputFile:
                is_test = True

            for sent_num, sentence in enumerate(data):
                text = sentence.text
                id2index = sentence._ids_to_indexes
                token_data, genre_token_data = None, None
                if self.args.use_wikid and genreparseddata_ and text in genreparseddata_:
                    genre_token_data = genreparseddata_[text]

                #Add the head-dependents
                dep_data_token = defaultdict(list)
                for token_num, token in enumerate(sentence):
                    dep_data_token[token.head].append(token.id)

                for token_num, token in enumerate(sentence):
                    token_id = token.id
                    if ("-" in token_id or "." in token_id):
                        continue
                    if not self.isValid(token.deprel):
                        continue
                    if token.head == '0': #current token is the root so no direction of left/right
                        continue
                    relation = token.deprel

                    pos = token.upos
                    feats = token.feats
                    lemma = token.lemma

                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    head_lemma = sentence[token.head].lemma

                    model_features = self.isValidFeature(pos, head_pos, relation)
                    if len(model_features) == 0:
                        #No relevant data found:
                        continue

                    token_position = id2index[token.id]
                    head_position = id2index[token.head]
                    if token_position < head_position:
                        label = 'before'
                    else:
                        label = 'after'

                    for model, exact_features in model_features.items():
                        if model not in self.model_dictionary:
                            self.model_dictionary[model] = len(self.model_dictionary)
                            self.model_data_case[model] = defaultdict(lambda: 0)
                            self.feature_freq[model] = defaultdict(lambda : 0)

                        feature = f'headpos_{head_pos}'
                        if not is_test:
                            self.feature_freq[model][feature] += 1

                        feature = f'deppos_{pos}'
                        if not is_test:
                            self.feature_freq[model][feature] += 1

                        feature = f'deprel_{relation}'
                        if not is_test:
                            self.feature_freq[model][feature] += 1

                        if not self.args.only_triples:
                            # Adding features for dependent token (maybe more commonly occurring)
                            for feat in feats:
                                if feat in self.remove_features:
                                    continue
                                feature = f'depfeat_{feat}_'
                                value = self.getFeatureValue(feat, feats)
                                feature += f'{value}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1

                            # Adding features for dependent token (maybe more commonly occurring)
                            for feat in head_feats:
                                if feat in self.remove_features:
                                    continue
                                feature = f'headfeat_{feat}_'
                                value = self.getFeatureValue(feat, head_feats)
                                feature += f'{value}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1

                            #Adding dep relation for the head
                            headrelation = sentence[token.head].deprel
                            if headrelation and headrelation != 'root' and headrelation != 'punct':
                                feature = f'headrelrel_{headrelation.lower()}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1

                            #Adding lemma of the head's head
                            if self.args.lexical:
                                headheadhead = sentence[token.head].head
                                if headheadhead != '0':
                                    headheadheadlemma = self.isValidLemma(sentence[headheadhead].lemma, sentence[headheadhead].upos)
                                    if headheadheadlemma:
                                        feature = f'headheadlemma_{headheadheadlemma}'
                                        if not is_test:
                                            self.feature_freq[model][feature] += 1

                            # get other dep tokens of the head
                            dep = dep_data_token.get(token.head, [])
                            for d in dep:
                                if d == token.id:
                                    continue
                                depdeprelation = sentence[d].deprel
                                if depdeprelation and depdeprelation != 'punct':
                                    feature = f'depheadrel_{depdeprelation}'
                                    if not is_test:
                                        self.feature_freq[model][feature] += 1

                                depdeppos = sentence[d].upos
                                if depdeppos in ['PUNCT', 'SYM', 'X']:
                                    continue
                                feature = f'depheadpos_{depdeppos}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1

                                depdeplemma = self.isValidLemma(sentence[d].lemma, sentence[d].upos)
                                if depdeplemma and self.args.lexical:
                                    feature = f'depheadlemma_{depdeplemma.lower()}'
                                    if not is_test:
                                        self.feature_freq[model][feature] += 1

                        if self.args.use_wikid:
                            # Get the wikidata features
                            if genre_token_data:
                                if token_id in genre_token_data:
                                    qids_all  = genre_token_data[token_id]
                                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                                    for feature in features:
                                        if not is_test:
                                            feature = 'wiki_' + feature
                                            self.feature_freq[model][feature] += 1

                                if token.head in genre_token_data:
                                    head_id = token.head
                                    qids_all = genre_token_data[head_id]
                                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                                    for feature in features:
                                        feature = 'wiki_head_' + feature
                                        if not is_test:
                                            self.feature_freq[model][feature] += 1

                        if self.args.lexical:
                            lemma = self.isValidLemma(lemma, pos)
                            if lemma:
                                feature = f'lemma_{lemma}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1


                            head_lemma = self.isValidLemma(head_lemma, head_pos)
                            if head_lemma:
                                feature = f'headlemma_{head_lemma}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1

                            # Add tokens in the neighborhood of 3
                            neighboring_tokens_left = max(0, token_num - 3)
                            neighboring_tokens_right = min(token_num + 3, len(sentence))
                            for neighor in range(neighboring_tokens_left, neighboring_tokens_right):
                                if neighor == token_num and neighor >= len(sentence):
                                    continue
                                neighor_token = sentence[neighor]
                                if neighor_token:
                                    lemma = self.isValidLemma(neighor_token.lemma, neighor_token.upos)
                                    if lemma:
                                        feature = f'neighborhood_{lemma}'
                                        if not is_test:
                                            self.feature_freq[model][feature] += 1

                        if self.args.use_spine:
                            vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim , token.lemma)
                            feature_names = spine_features[vector > 0] #Get active features
                            for feature in feature_names:
                                feature = f'spine_{feature}'
                                if not is_test:
                                    self.feature_freq[model][feature] += 1

                            if token.head and token.head != '0':
                                vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim,
                                                               sentence[token.head].lemma)
                                feature_names = spine_features[vector > 0]  # Get active features
                                for feature in feature_names:
                                    feature = f'spinehead_{feature}'
                                    if not is_test:
                                        self.feature_freq[model][feature] += 1

                        self.model_data[model] += 1  # Dependent of pos
                        self.model_data_case[model][label] += 1


        self.id2model = {v: k for k, v in self.model_dictionary.items()}
        self.feature_map_id2model = {}
        with open(vocab_file, 'w') as fout:
            for model, items in self.feature_freq.items():
                self.feature_map[model] = {}
                for feature, freq in items.items():
                    if freq < 50:
                        continue
                    self.feature_map[model][feature] = len(self.feature_map[model])
                self.feature_map_id2model[model] = {v:k for k,v in  self.feature_map[model].items()}
                fout.write(f'Model:{model}\n')
                for k,v in self.feature_map[model].items():
                    fout.write(f'{v}\t{k}\n')

                labels = self.model_data_case[model]
                fout.write(f'Data:{self.model_data[model]}\tLabels\t')
                label_values = []
                for label, v in labels.items():
                    label_values.append(label + "," + str(v))
                fout.write(";".join(label_values) + "\n")

            fout.write('\n')

    def getFeatureValue(self, feat, feats):
        values = list(feats[feat])
        values.sort()
        value = "/".join(values)
        return value

    def getWikiDataParse(self, babel_ids, wikidata):
        features = []
        for babel_id in babel_ids:
            if babel_id in wikidata:
                wikifeatures = wikidata[babel_id]
                for wiki, _ in wikifeatures.items():
                    feature = f'{wiki}'
                    features.append(feature)
        return features

    def getWikiDataGenreParse(self, qids_all, wikidata):
        features = []
        for qids in qids_all: #qids = [Q12, Q23, ...]
            for qid in qids.split(","):
                if qid in wikidata:
                    wikifeatures = wikidata[qid]
                    for wiki, _ in wikifeatures.items():
                        feature = f'wiki_{wiki}'
                        features.append(feature)
                    break
        return features

    def getSpineFeatures(self, spine_word_vectors, word, dim, lemma):
        vector = [0 for _ in range(dim)]
        if word.lower() in spine_word_vectors:
            vector = spine_word_vectors[word.lower()]
        #     if lemma and lemma.lower() not in spine_word_vectors:
        #         spine_word_vectors[lemma.lower()] = vector
        # elif lemma and lemma.lower() in spine_word_vectors:
        #     vector = spine_word_vectors[lemma.lower()]
        return np.array(vector)

    def isValid(self,relation):
        found = False
        if not relation:
            return found
        relation = relation.lower()
        if len(self.required_relations) > 0:  # Restrict analysis to relations
            for rel in self.required_relations:
                if rel in relation:
                    found = True
        else:
            found = True
        return found

    def addFeature(self, model, feature, feature_array, label, value = 1):
        feature_id = self.feature_map[model].get(feature, -1)
        if feature_id >= 0:
            feature_array[feature_id] = value
            self.model_feature_label[model][label][feature] += 1

    def getFeatures(self, inputFile,
                                  model,
                                  labels,
                                  genreparseddata,
                                  wikidata,
                                spine_word_vectors,
                                spine_features,
                                spine_dim,
                                filename):
        if model not in self.model_feature_label:
            self.model_feature_label[model] = {}
        f = inputFile.strip()
        data = pyconll.load_from_file(f"{f}")
        all_features = []

        columns = []
        output_labels = defaultdict(lambda : 0)
        num_of_tokens_syntax = 0
        num_of_tokens_wikigenre = 0
        num_of_tokens_lexical = 0
        num_of_tokens_spine = 0

        index = [i for i in range(len(data))]
        #Get column names
        for i in range(len(self.feature_map[model])):
            feature_name = self.feature_map_id2model[model][i]
            columns.append(feature_name)
        columns.append('label')
        columns.append('sent_num')
        columns.append('token_num')
        num = 0
        for sentence_num in index:
            sentence = data[sentence_num]
            text = sentence.text
            id2index = sentence._ids_to_indexes
            token_data, genre_token_data = None, None
            if self.args.use_wikid and genreparseddata and text in genreparseddata:
                genre_token_data = genreparseddata[text]

            dep_data_token = defaultdict(list)
            for token_num, token in enumerate(sentence):
                dep_data_token[token.head].append(token.id)

            for token_num, token in enumerate(sentence):
                token_id = token.id
                if ("-" in token_id or "." in token_id):
                    continue
                if not self.isValid(token.deprel):
                    continue
                if token.head == '0':  # current token is the root so no direction of left/right
                    continue

                relation = token.deprel.lower()
                pos = token.upos
                feats = token.feats
                lemma = token.lemma
                head_pos = sentence[token.head].upos
                head_feats = sentence[token.head].feats
                head_lemma = sentence[token.head].lemma

                model_features = self.isValidFeature(pos, head_pos, relation)
                if len(model_features) == 0 or model not in model_features:
                    # No relevant data found:
                    continue


                num_of_tokens_syntax += 1
                feature_array = np.zeros((len(self.feature_map[model]),), dtype=float)

                token_position = id2index[token.id]
                head_position = id2index[token.head]
                if token_position < head_position:
                    label = 'before'
                else:
                    label = 'after'

                if label not in self.model_feature_label[model]:
                    self.model_feature_label[model][label] = defaultdict(lambda : 0)

                feature = f'headpos_{head_pos}'
                self.addFeature(model, feature, feature_array, label)

                feature = f'deppos_{pos}'
                self.addFeature(model, feature, feature_array, label)

                feature = f'deprel_{relation}'
                self.addFeature(model, feature, feature_array, label)

                if not self.args.only_triples:
                    for feat in feats:  # Adding features for dependent token (maybe more commonly occurring)
                        #if feat in feats:
                        feature = f'depfeat_{feat}_'
                        value = self.getFeatureValue(feat, feats)
                        feature += f'{value}'
                        self.addFeature(model, feature, feature_array, label)

                    for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
                        #if feat in head_feats:
                        feature = f'headfeat_{feat}_'
                        value = self.getFeatureValue(feat, head_feats)
                        feature += f'{value}'
                        self.addFeature(model, feature, feature_array, label)

                    headrelation = sentence[token.head].deprel
                    if headrelation and headrelation != 'root':
                        feature = f'headrelrel_{headrelation}'
                        self.addFeature(model, feature, feature_array, label)

                    if self.args.lexical:
                        headheadhead = sentence[token.head].head
                        if headheadhead != '0':
                            headheadheadlemma = self.isValidLemma(sentence[headheadhead].lemma, sentence[headheadhead].upos)
                            if headheadheadlemma:
                                feature = f'headheadlemma_{headheadheadlemma}'
                                self.addFeature(model, feature, feature_array, label)

                    dep = dep_data_token.get(token.head, [])
                    for d in dep:
                        if d == token.id:
                            continue
                        depdeprelation = sentence[d].deprel
                        if depdeprelation:
                            #get the deprel of other deps of the head
                            feature = f'depheadrel_{depdeprelation}'
                            self.addFeature(model, feature, feature_array, label)

                        depdeppos = sentence[d].upos
                        feature = f'depheadpos_{depdeppos}'
                        self.addFeature(model, feature, feature_array, label)

                        depdeplemma = self.isValidLemma(sentence[d].lemma, sentence[d].upos)
                        if depdeplemma:
                            feature = f'depheadlemma_{depdeplemma}'
                            self.addFeature(model, feature, feature_array, label)

                if self.args.use_wikid:
                    # Get the wikidata features
                    if genre_token_data:
                        isWiki = False
                        if token_id in genre_token_data:
                            qids_all = genre_token_data[token_id]
                            features = self.getWikiDataGenreParse(qids_all, wikidata)
                            for feature in features:
                                feature = 'wiki_' + feature
                                self.addFeature(model, feature, feature_array, label)
                                isWiki = True
                            if isWiki:
                                num_of_tokens_wikigenre += 1

                        if token.head in genre_token_data:
                            head_id = token.head
                            qids_all = genre_token_data[head_id]
                            features = self.getWikiDataGenreParse(qids_all, wikidata)
                            for feature in features:
                                feature = 'wiki_head_' + feature
                                self.addFeature(model, feature, feature_array, label)

                if self.args.lexical:

                    lemma = self.isValidLemma(lemma, pos)
                    if lemma:
                        num_of_tokens_lexical += 1
                        feature = f'lemma_{lemma}'
                        self.addFeature(model, feature, feature_array, label)

                    head_lemma = self.isValidLemma(head_lemma, head_pos)
                    if head_lemma:
                        feature = f'headlemma_{head_lemma}'
                        self.addFeature(model, feature, feature_array, label)

                    #Add tokens in the neighborhood of 3
                    neighboring_tokens_left = max(0, token_num - 3)
                    neighboring_tokens_right = min(token_num + 3, len(sentence))
                    for neighor in range(neighboring_tokens_left, neighboring_tokens_right):
                        if neighor == token_num and neighor < len(sentence):
                            continue
                        #print(neighor, len(sentence))
                        neighor_token = sentence[neighor]
                        if neighor_token and neighor_token.lemma:
                            lemma = self.isValidLemma(neighor_token.lemma, neighor_token.upos)
                            if lemma:
                                feature = f'neighborhood_{lemma}'
                                self.addFeature(model, feature, feature_array, label)

                if self.args.use_spine:
                    isspine = False
                    vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                    feature_names = spine_features[vector > 0]  # Get active features
                    vector_filtered = vector[vector > 0]
                    for feature, value in zip(feature_names, vector_filtered):
                        feature = f'spine_{feature}'
                        self.addFeature(model, feature, feature_array, label, value=value)
                        isspine = True
                    if isspine:
                        num_of_tokens_spine += 1

                    if token.head and token.head != '0':
                        vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim,
                                                       sentence[token.head].lemma)
                        feature_names = spine_features[vector > 0]  # Get active features
                        vector_filtered = vector[vector > 0]
                        for feature, value in zip(feature_names, vector_filtered):
                            feature = f'spinehead_{feature}'
                            self.addFeature(model, feature, feature_array, label, value=value)

                one_feature = np.concatenate(
                    (feature_array, label, sentence_num, token.id), axis=None)
                assert len(one_feature) == len(columns)
                all_features.append(one_feature)
                output_labels[label] += 1
                num += 1
                #df = pd.DataFrame([one_feature], columns=columns)
                #df.to_csv(filename, mode='a', header=not os.path.exists(filename))

        print(f'Syntax: {num_of_tokens_syntax}  Wikigenre: {num_of_tokens_wikigenre}, Lexical: {num_of_tokens_lexical}, Spine:{num_of_tokens_spine} columns: {len(columns)}')
        random.shuffle(all_features)

        return all_features, columns, output_labels


    def getHistogram(self, filename, input_path, feature):
        f = input_path.strip()
        data = pyconll.load_from_file(f"{f}")
        self.feature_tokens, self.feature_forms = {}, {}
        self.feature_tokens[feature] = defaultdict(lambda: 0)
        self.feature_forms = {}
        self.feature_forms_num = {}
        tokens, feature_values, pos_values= [], [], []
        self.lemma, self.lemmaGroups, self.lemma_freq, self.lemma_inflection = {}, defaultdict(set), {}, {}
        pos_barplots = {}
        features_set, pos_count = set(), defaultdict(lambda : 0)
        label_value = defaultdict(lambda : 0)

        for sentence in data:
            for token in sentence:
                if token.form == None or "-" in token.id:
                    continue

                token_id = token.id
                relation = token.deprel
                pos = token.upos
                if pos == None:
                    pos = 'None'
                feats = token.feats
                lemma = token.lemma

                tokens.append(token.form)
                pos_values.append(pos)

                pos_count[pos] += 1
                self.lemma[token.form.lower()] = lemma
                self.lemmaGroups[lemma].add(token.form.lower())
                if pos not in self.feature_forms_num:
                    self.feature_forms_num[pos] = {}
                    pos_barplots[pos] = defaultdict(lambda : 0)

                if pos not in self.lemma_inflection:
                    self.lemma_freq[pos] = defaultdict(lambda: 0)
                    self.lemma_inflection[pos] = {}

                if lemma:
                    self.lemma_freq[pos][lemma.lower()] += 1
                if lemma and lemma.lower() not in self.lemma_inflection[pos]:
                    self.lemma_inflection[pos][lemma.lower()] = {}
                # Aggregae morphology properties of required-properties - feature
                morphology_props = set(self.args.features) - set([feature])
                morphology_prop_values = []
                for morphology_prop in morphology_props:
                    if morphology_prop in feats:
                        morphology_prop_values.append(",".join(feats[morphology_prop]))
                morphology_prop_values.sort()
                inflection = ";".join(morphology_prop_values)
                if lemma and inflection not in self.lemma_inflection[pos][lemma.lower()]:
                    self.lemma_inflection[pos][lemma.lower()][inflection] = {}
                if feature in feats:
                    values = list(feats[feature])
                    values.sort()
                    feature_values.append("/".join(values))
                    label_value["/".join(values)] += 1
                    #for feat in values:
                else:
                    values = ['NA']
                    feature_values.append("NA")
                    label_value['NA'] += 1
                feat = "/".join(values)
                features_set.add(feat)
                pos_barplots[pos][feat] += 1
                if feat not in self.feature_forms_num[pos]:
                    self.feature_forms_num[pos][feat] = defaultdict(lambda : 0)

                self.feature_forms_num[pos][feat][token.form.lower()] += 1
                if lemma:
                    self.lemma_inflection[pos][lemma.lower()][inflection][feat] = token.form.lower()

        #sort the pos by count
        sorted_pos = sorted(pos_count.items(), key= lambda kv: kv[1], reverse=True)
        pos_to_id, pos_order = {}, []
        for (pos, _) in sorted_pos:
            pos_to_id[pos] = len(pos_to_id)
            pos_order.append(pos)

        #Stacked histogram
        #sns.set()
        fig, ax = plt.subplots()
        bars_num = np.zeros((len(features_set), len(pos_barplots)))
        x_axis = []
        feat_to_id, id_to_feat = get_vocab_from_set(features_set)

        for pos in pos_order:
            feats = pos_barplots[pos]
            x_axis.append(pos)
            pos_id = pos_to_id[pos]
            for feat, num in feats.items():
                feat_id = feat_to_id[feat]
                bars_num[feat_id][pos_id] = num

        r = [i for i in range(len(pos_to_id))]
        handles, color = [], ['steelblue', 'orange', 'olivedrab', 'peru', 'seagreen', 'chocolate',
                              'tan', 'lightseagreen', 'green', 'teal','tomato','lightgreen','yellow','lightblue','azure','red',
                              'aqua', 'darkgreen', 'tomato', 'firebrick', 'khaki', 'gold', 'powderblue',  'navy', 'plum' ]
        bars = np.zeros((len(pos_barplots)))
        handle_map = {}
        for barnum in range(len(features_set)):
            plt.bar(r, bars_num[barnum], bottom=bars, color=color[barnum], edgecolor='white', width=1)
            handle_map[id_to_feat[barnum]]= mpatches.Patch(color=color[barnum], label=id_to_feat[barnum])
            bars += bars_num[barnum]

        #Sort legend by frequency
        sorted_labels = sorted(label_value.items(), key=lambda kv:kv[1], reverse=True)
        for (label, _) in sorted_labels:
            handles.append(handle_map[label])

        #handles.reverse()
        # Custom X axis
        plt.xticks(r, x_axis,rotation=45, fontsize=9)
        #plt.xlabel("pos")
        plt.ylabel("Number of Tokens")
        plt.legend(handles=handles)

        right_side = ax.spines["right"]
        right_side.set_visible(False)

        top_side = ax.spines["top"]
        top_side.set_visible(False)

        plt.savefig(f"{filename}" + "/pos.png", transparent=True)
        plt.close()
        return pos_order

    def example_web_print(self, ex, outp2, data):
        try:
            # print('\t\t',data[ex[0]].text)
            sentid = int(ex[0])
            tokid = str(ex[1])
            active  = ex[2]
            req_head_head = False
            req_dep_dep = False
            req_dep_head = False
            for feature in active:
                if feature.startswith("headrelrel"):
                    req_head_head = True

                if feature.startswith('dephead'):
                    req_dep_dep = True

                info = feature.split("_")
                if len(info) > 2:
                    if feature.startswith("agree"):
                        req_head_head = True

            headid = data[sentid][tokid].head
            outp2.write('<pre><code class="language-conllu">\n')
            dep_data_token = defaultdict(list)
            for token in data[sentid]:
                dep_data_token[token.id] = token.head
            already_covered = set()
            for token in data[sentid]:
                if token.id in already_covered:
                    continue
                if token.id == tokid:
                    #print the original dependent token
                    temp = token.conll().split('\t')
                    temp[1] = "***" + temp[1] + "***"
                    temp[8] = '_'
                    temp2 = '\t'.join(temp)
                    outp2.write(temp2 + "\n")
                    already_covered.add(token.id)
                elif token.id == headid:
                    #print the head of the original token
                    if req_head_head:
                        temp = token.conll().split('\t')
                        temp[8] = '_'
                        temp2 = '\t'.join(temp)
                        outp2.write(temp2 + "\n")
                    else:
                        temp = token.conll().split('\t')
                        temp2 = '\t'.join(temp[:6])
                        outp2.write(f"{temp2}\t0\t_\t_\t_\n")
                    already_covered.add(token.id)

                elif token.id == data[sentid][headid].head and req_head_head:
                    temp = token.conll().split('\t')
                    temp2 = '\t'.join(temp[:6])
                    outp2.write(f"{temp2}\t0\t_\t_\t_\n")
                    already_covered.add(token.id)

                elif token.id in dep_data_token and dep_data_token[token.id] == headid and req_dep_dep:
                    #this token is also dependent of the head-token
                    temp = token.conll().split('\t')
                    temp[8] = '_'
                    temp2 = '\t'.join(temp)
                    outp2.write(temp2 + "\n")
                    already_covered.add(token.id)

                elif token.id in dep_data_token and dep_data_token[token.id] == tokid and req_dep_dep:
                    #this token is dependent of the original token
                    temp = token.conll().split('\t')
                    temp[8] = '_'
                    temp2 = '\t'.join(temp)
                    outp2.write(temp2 + "\n")
                    already_covered.add(token.id)

                elif '-' not in token.id:
                    outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
                    already_covered.add(token.id)

            outp2.write('\n</code></pre>\n\n')
        except:
            pass

    def addFeaturesHead(self, token):
        features = []

        for feat in self.required_features:  # Adding features for dependent token (maybe more commonly occurring)
            feats = token.feats
            if feat in feats:
                feature = f'{feat}_'
                value = self.getFeatureValue(feat, feats)
                feature += f'{value}'
                features.append(feature)
        return features


