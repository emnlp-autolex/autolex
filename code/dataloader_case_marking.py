import pyconll
import numpy as np
np.random.seed(1)
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import random


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
        self.pos_dictionary = {}
        self.feature_dictionary = {}
        self.relation_dictionary = {}
        self.used_relations = set()
        self.used_head_pos = set()
        self.used_child_pos = set()
        self.used_tuples = defaultdict(set)
        self.class_relations = defaultdict(set)
        self.class_headpos = defaultdict(set)
        self.class_childpos = defaultdict(set)
        self.relation_map = relation_map
        self.feature_map = {}
        self.triple_freq = defaultdict(lambda: 0)
        self.pos_data = defaultdict(lambda : 0)
        self.pos_data_case = {}
        self.feature_map_pos = {}
        self.feature_freq = {}
        self.required_features = [] #'Tense', 'Animacy', 'Aspect', 'Mood', 'Definite', 'Voice', 'Poss','PronType', 'VerbForm', 'Person', 'Polarity']
        self.required_relations = [] #['obj', 'iobj', 'amod', 'det', 'mod', 'nsubj', 'obl', 'vocative', 'aux', 'compound', 'conj']  # ,
        self.remove_features = ['Gender', 'Person', 'Number']
        random.seed(args.seed)

    def unison_shuffled_copies(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]


    def isValidLemma(self, lemma, upos):
        if upos == 'PUNCT' or upos == 'NUM':
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

            for sentence in data:
                text = sentence.text
                id2index = sentence._ids_to_indexes
                token_data, genre_token_data = None, None
                if self.args.use_wikid and genreparseddata_ and text in genreparseddata_:
                    genre_token_data = genreparseddata_[text]

                dep_data_token = defaultdict(list)
                for token_num, token in enumerate(sentence):
                    dep_data_token[token.head].append(token.id)

                for token_num, token in enumerate(sentence):
                    token_id = token.id
                    if ("-" in token_id or "." in token_id):
                        continue
                    if not self.isValid(token.deprel):
                        continue
                    relation = token.deprel
                    pos = token.upos
                    feats = token.feats
                    lemma = token.lemma

                    if 'Case'not in feats:
                        continue
                    label = self.getFeatureValue('Case', feats)

                    if token.head and token.head != "0" and pos and relation:
                        if pos not in self.pos_dictionary:
                            self.pos_dictionary[pos] = len(self.pos_dictionary)
                            self.pos_data_case[pos] = defaultdict(lambda: 0)
                            self.feature_freq[pos] = defaultdict(lambda : 0)

                        head_pos = sentence[token.head].upos
                        head_feats = sentence[token.head].feats
                        headrelation = sentence[token.head].deprel
                        head_lemma = sentence[token.head].lemma

                        feature = f'headpos_{head_pos}'
                        if not is_test:
                            self.feature_freq[pos][feature] += 1


                        feature = f'deprel_{relation}'
                        if not is_test:
                            self.feature_freq[pos][feature] += 1

                        if not self.args.only_triples:
                            if 'Case' in head_feats:
                                headlabel = self.getFeatureValue('Case', head_feats)

                                if label == headlabel: #If agreement between the head-dep
                                    feature = f'agreepos_{head_pos}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                    feature = f'agreerel_{relation}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                    if headrelation:
                                        headrelation = headrelation.lower()
                                        if  headrelation != 'root' and headrelation != 'punct' and "unk" not in headrelation:
                                            feature = f'agree_{relation}_{head_pos}_{headrelation.lower()}'
                                            if not is_test:
                                                self.feature_freq[pos][feature] += 1

                                            feature = f'agree_{headrelation.lower()}'
                                            if not is_test:
                                                self.feature_freq[pos][feature] += 1

                            for feat in feats: #self.required_features:  # Adding features for dependent token (maybe more commonly occurring)
                                if feat in self.remove_features or feat == 'Case':
                                    continue
                                feature = f'depfeat_{feat}_'
                                value = self.getFeatureValue(feat, feats)
                                feature += f'{value}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                            for feat in head_feats:  # self.required_features:  # Adding features for dependent token (maybe more commonly occurring)
                                if feat in self.remove_features:
                                    continue
                                feature = f'headfeat_{head_pos}_{relation}_{feat}_'
                                value = self.getFeatureValue(feat, head_feats)
                                feature += f'{value}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                                feature = f'headfeat_{head_pos}_{feat}_{value}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                                feature = f'headfeatrel_{relation}_{feat}_{value}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                                feature = f'headfeat_{feat}_{value}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                            if headrelation:
                                headrelation = headrelation.lower()
                                if headrelation != 'root' and headrelation != 'punct' and "unk" not in headrelation:
                                    feature = f'headrelrel_{head_pos}_{relation}_{headrelation.lower()}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                    feature = f'headrelrel_{head_pos}_{headrelation.lower()}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                    feature = f'headrelrel_{headrelation.lower()}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                            # adding the children of the dep token
                            for dep in dep_data_token[token.id]:
                                deptoken = sentence[dep]
                                feature = f'depdeppos_{deptoken.upos}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                                deprel = deptoken.deprel
                                if deprel:
                                    deprel = deprel.lower()
                                    if deprel != 'root' and deprel != 'punct' and "unk" not in deprel:

                                        feature = f'depdeprel_{deprel}'
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                                deplemma = self.isValidLemma(deptoken.lemma, deptoken.upos)
                                if self.args.lexical and deplemma:
                                    feature = f'depdeplemma_{deplemma}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                            # get other dep tokens of the head
                            dep = dep_data_token.get(token.head, [])
                            for d in dep:
                                if d == token.id:
                                    continue
                                depdeprelation = sentence[d].deprel
                                if depdeprelation:
                                    depdeprelation = depdeprelation.lower()
                                    if depdeprelation != 'root' and depdeprelation != 'punct' and 'unk' not in depdeprelation:
                                        feature = f'depheadrel_{depdeprelation}'
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                                depdeppos = sentence[d].upos
                                feature = f'depheadpos_{depdeppos}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                                depdeplemma = self.isValidLemma(sentence[d].lemma, sentence[d].upos)
                                if self.args.lexical and depdeplemma:
                                    feature = f'depheadlemma_{depdeplemma.lower()}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                        # Adding lemma of the head's head
                        if self.args.lexical:
                            lemma = self.isValidLemma(lemma, pos)
                            if lemma:
                                feature = f'lemma_{lemma}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                            head_lemma = self.isValidLemma(head_lemma, head_pos)
                            if head_lemma:
                                feature = f'headlemma_{head_lemma}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

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
                                            self.feature_freq[pos][feature] += 1

                            headheadhead = sentence[token.head].head
                            if headheadhead != '0':
                                headheadheadlemma = self.isValidLemma(sentence[headheadhead].lemma,
                                                                      sentence[headheadhead].upos)
                                if headheadheadlemma:
                                    feature = f'headheadlemma_{headheadheadlemma}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                        if self.args.use_wikid:
                            # Get the wikidata features
                            if genre_token_data and token_id in genre_token_data:
                                qids_all  = genre_token_data[token_id]
                                features = self.getWikiDataGenreParse(qids_all, wikidata)
                                for feature in features:
                                    if not is_test:
                                        feature = 'wiki_' + feature
                                        self.feature_freq[pos][feature] += 1

                            if genre_token_data and token.head in genre_token_data:
                                qids_all = genre_token_data[token.head]
                                features = self.getWikiDataGenreParse(qids_all, wikidata)
                                for feature in features:
                                    feature = 'wiki_head_' + feature
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                        if self.args.use_spine:
                            vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                            feature_names = spine_features[vector == 1] #Get active features
                            for feature in feature_names:
                                feature = f'spine_{feature}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                            if token.head and token.head != '0':
                                vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim,
                                                               sentence[token.head].lemma)
                                feature_names = spine_features[vector == 1]  # Get active features
                                for feature in feature_names:
                                    feature = f'spinehead_{feature}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                        self.pos_data[pos] += 1  # Dependent of pos
                        self.pos_data_case[pos][label] += 1

        self.pos_id2tag = {v: k for k, v in self.pos_dictionary.items()}
        self.feature_map_pos_id2tag = {}
        with open(vocab_file, 'w') as fout:

            for pos, items in self.feature_freq.items():
                self.feature_map_pos[pos] = {}
                for feature, freq in items.items():
                    if freq < 50:
                        continue
                    self.feature_map_pos[pos][feature] = len(self.feature_map_pos[pos])
                self.feature_map_pos_id2tag[pos] = {v:k for k,v in  self.feature_map_pos[pos].items()}
                fout.write(f'POS:{pos}\n')
                for k,v in self.feature_map_pos[pos].items():
                    fout.write(f'{v}\t{k}\n')

                labels = self.pos_data_case[pos]
                fout.write(f'Data:{self.pos_data[pos]}\tLabels\t')
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
                    feature = f'wiki_{wiki}'
                    features.append(feature)
        return features

    def getWikiDataGenreParse(self, qids_all, wikidata):
        features = []
        for qids in qids_all: #qids = [Q12, Q23, ...]
            for qid in qids.split(","):
                if qid in wikidata:
                    wikifeatures = wikidata[qid]
                    for wiki, _ in wikifeatures.items():
                        feature = f'{wiki}'
                        features.append(feature)
                    break
        return features


    def getSpineFeatures(self, spine_word_vectors, word, dim, lemma):
        vector = [0 for _ in range(dim)]
        if word.lower() in spine_word_vectors:
            vector = spine_word_vectors[word.lower()]

        return np.array(vector)


    def isValid(self,relation):
        found = False
        if not relation:
            return found
        relation = relation.lower()
        if "unk" in relation:
            return found
        if len(self.required_relations) > 0:  # Restrict analysis to relations
            for rel in self.required_relations:
                if rel in relation:
                    found = True
        else:
            found = True
        return found


    def addFeature(self, model, feature, feature_array, label):
        feature_id = self.feature_map_pos[model].get(feature, -1)
        if feature_id >= 0:
            feature_array[feature_id] = 1

    def getCaseAssignmentFeatures(self, inputFile,
                                  pos,
                                  labels,
                                  wikiparseddata,
                                  genreparseddata,
                                  wikidata,
                                  spine_word_vectors,
                                  spine_features,
                                  spine_dim, filename ):
        f = inputFile.strip()
        data = pyconll.load_from_file(f"{f}")
        all_features = []
        columns = []
        output_labels = defaultdict(lambda : 0)


        num_of_tokens_wiki = 0
        num_of_tokens_syntax = 0
        num_of_tokens_lexical = 0
        num_of_tokens_spine = 0
        num = 0

        index = [i for i in range(len(data))]
        #Get column names
        for i in range(len(self.feature_map_pos[pos])):
            feature_name = self.feature_map_pos_id2tag[pos][i]
            columns.append(feature_name)
        columns.append('label')
        columns.append('sent_num')
        columns.append('token_num')

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
                if ("-"in token_id or "." in token_id):
                    continue

                relation = token.deprel
                feats = token.feats
                lemma = token.lemma
                if relation == 'unk' or token.upos != pos or 'Case' not in feats:
                    continue

                label = self.getFeatureValue('Case', feats)
                if label not in labels:
                    continue

                feature_array = np.zeros((len(self.feature_map_pos[pos]),), dtype=int)
                if token.head and token.head != "0" and pos:
                    num_of_tokens_syntax += 1

                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    headrelation = sentence[token.head].deprel
                    head_lemma = sentence[token.head].lemma

                    feature = f'headpos_{head_pos}'
                    self.addFeature(pos, feature, feature_array, label)

                    if self.isValid(relation):
                        feature = f'deprel_{relation}'

                        self.addFeature(pos, feature, feature_array, label)

                    if not self.args.only_triples:
                        if 'Case' in head_feats:
                            headlabel = self.getFeatureValue('Case', head_feats)

                            if label == headlabel:  # If agreement between the head-dep
                                feature = f'agreepos_{head_pos}'
                                self.addFeature(pos, feature, feature_array, label)

                                feature = f'agreerel_{relation}'
                                self.addFeature(pos, feature, feature_array, label)

                                if headrelation and headrelation != 'root' and headrelation != 'punct':
                                    feature = f'agree_{relation}_{head_pos}_{headrelation.lower()}'
                                    self.addFeature(pos, feature, feature_array, label)

                                    feature = f'agree_{headrelation.lower()}'
                                    self.addFeature(pos, feature, feature_array, label)

                        for feat in feats:  # self.required_features:  # Adding features for dependent token (maybe more commonly occurring)
                            if feat in self.remove_features:
                                continue
                            feature = f'depfeat_{feat}_'
                            value = self.getFeatureValue(feat, feats)
                            feature += f'{value}'
                            self.addFeature(pos, feature, feature_array, label)

                        for feat in head_feats:  # self.required_features:  # Adding features for dependent token (maybe more commonly occurring)
                            if feat in self.remove_features:
                                continue
                            feature = f'headfeat_{head_pos}_{relation}_{feat}_'
                            value = self.getFeatureValue(feat, head_feats)
                            feature += f'{value}'
                            self.addFeature(pos, feature, feature_array, label)

                            feature = f'headfeat_{head_pos}_{feat}_{value}'
                            self.addFeature(pos, feature, feature_array, label)

                            feature = f'headfeatrel_{relation}_{feat}_{value}'
                            self.addFeature(pos, feature, feature_array, label)

                            feature = f'headfeat_{feat}_{value}'
                            self.addFeature(pos, feature, feature_array, label)

                        if headrelation and headrelation != 'root' and headrelation != 'punct':
                            feature = f'headrelrel_{head_pos}_{relation}_{headrelation.lower()}'
                            self.addFeature(pos, feature, feature_array, label)

                            feature = f'headrelrel_{head_pos}_{headrelation.lower()}'
                            self.addFeature(pos, feature, feature_array, label)

                            feature = f'headrelrel_{headrelation.lower()}'
                            self.addFeature(pos, feature, feature_array, label)

                        # adding the children of the dep token
                        for dep in dep_data_token[token.id]:
                            deptoken = sentence[dep]
                            feature = f'depdeppos_{deptoken.upos}'
                            self.addFeature(pos, feature, feature_array, label)

                            deprel = deptoken.deprel
                            if deprel and deprel != 'root' and deprel != 'punct':
                                feature = f'depdeprel_{deprel}'
                                self.addFeature(pos, feature, feature_array, label)

                            deplemma = self.isValidLemma(deptoken.lemma, deptoken.upos)
                            if self.args.lexical and deplemma:
                                feature = f'depdeplemma_{deplemma}'
                                self.addFeature(pos, feature, feature_array, label)

                        # get other dep tokens of the head
                        dep = dep_data_token.get(token.head, [])
                        for d in dep:
                            if d == token.id:
                                continue
                            depdeprelation = sentence[d].deprel
                            if depdeprelation and depdeprelation != 'punct':
                                feature = f'depheadrel_{depdeprelation}'
                                self.addFeature(pos, feature, feature_array, label)

                            depdeppos = sentence[d].upos
                            feature = f'depheadpos_{depdeppos}'
                            self.addFeature(pos, feature, feature_array, label)

                            depdeplemma = self.isValidLemma(sentence[d].lemma, sentence[d].upos)
                            if depdeplemma and self.args.lexical:
                                feature = f'depheadlemma_{depdeplemma}'
                                self.addFeature(pos, feature, feature_array, label)

                    # Adding lemma of the head's head
                    if self.args.lexical:
                        lemma = self.isValidLemma(lemma, pos)
                        if lemma:
                            num_of_tokens_lexical += 1
                            feature = f'lemma_{lemma}'
                            self.addFeature(pos, feature, feature_array, label)

                        head_lemma = self.isValidLemma(head_lemma, head_pos)
                        if head_lemma:
                            feature = f'headlemma_{head_lemma}'
                            self.addFeature(pos, feature, feature_array, label)

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
                                    num_of_tokens_lexical += 1
                                    feature = f'neighborhood_{lemma}'
                                    self.addFeature(pos, feature, feature_array, label)

                        headheadhead = sentence[token.head].head
                        if headheadhead != '0':
                            headheadheadlemma = self.isValidLemma(sentence[headheadhead].lemma,
                                                                  sentence[headheadhead].upos)
                            if headheadheadlemma:
                                feature = f'headheadlemma_{headheadheadlemma}'
                                self.addFeature(pos, feature, feature_array, label)

                    if self.args.use_wikid:
                        # Get the wikidata features
                        isWiki = False
                        if genre_token_data and token_id in genre_token_data:
                            qids_all = genre_token_data[token_id]
                            features = self.getWikiDataGenreParse(qids_all, wikidata)
                            for feature in features:
                                feature = 'wiki_' + feature
                                self.addFeature(pos, feature, feature_array, label)
                                isWiki = True
                            if isWiki:
                                num_of_tokens_wiki += 1

                        if genre_token_data and token.head in genre_token_data:
                            head_id = token.head
                            qids_all = genre_token_data[head_id]
                            features = self.getWikiDataGenreParse(qids_all, wikidata)
                            for feature in features:
                                feature = 'wiki_head_' + feature
                                self.addFeature(pos, feature, feature_array, label)

                    if self.args.use_spine:
                        isspine = False
                        vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                        feature_names = spine_features[vector == 1]  # Get active features
                        for feature in feature_names:
                            feature = f'spine_{feature}'
                            self.addFeature(pos, feature, feature_array, label)
                            isspine = True
                        if isspine:
                            num_of_tokens_spine += 1

                        if token.head and token.head != '0':
                            vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim,
                                                           sentence[token.head].lemma)
                            feature_names = spine_features[vector == 1]  # Get active features
                            for feature in feature_names:
                                feature = f'spinehead_{feature}'
                                self.addFeature(pos, feature, feature_array, label)

                    one_feature = np.concatenate(
                                (feature_array, label, sentence_num, token.id), axis=None)
                    assert len(one_feature) == len(columns)
                    all_features.append(one_feature)
                    output_labels[label] += 1
                    num += 1
                    #df = pd.DataFrame([one_feature], columns=columns)
                    #df.to_csv(filename, mode='a', header=not os.path.exists(filename))

        print(f' Syntax: {num_of_tokens_syntax}, Spine: {num_of_tokens_spine}, Lexical: {num_of_tokens_lexical}, Wiki: {num_of_tokens_wiki},  Columns: {len(columns)}')
        #random.shuffle(all_features)
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
            req_head_head = True
            for feature in active:
                info = feature.split("_")
                if len(info) > 2:
                    if feature.startswith("agree"):
                        req_head_head = True

            headid = data[sentid][tokid].head
            outp2.write('<pre><code class="language-conllu">\n')
            for token in data[sentid]:
                if token.id == tokid:
                    temp = token.conll().split('\t')
                    temp[1] = "***" + temp[1] + "***"
                    temp2 = '\t'.join(temp)
                    outp2.write(temp2 + "\n")
                elif token.id == headid:
                    if req_head_head:
                        outp2.write(token.conll() + "\n")
                    else:
                        temp = token.conll().split('\t')
                        temp2 = '\t'.join(temp[:6])
                        outp2.write(f"{temp2}\t0\t_\t_\t_\n")
                elif token.id == data[sentid][headid].head and req_head_head:
                    temp = token.conll().split('\t')
                    temp2 = '\t'.join(temp[:6])
                    outp2.write(f"{temp2}\t0\t_\t_\t_\n")
                elif '-' not in token.id:
                    outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
            outp2.write('\n</code></pre>\n\n')
        # print(f"\t\t\tID: {tokid} token: {data[sentid][tokid].form}\t{data[sentid][tokid].upos}\t{data[sentid][tokid].feats}")
        # print(data[sentid][tokid].conll())
        # headid = data[sentid][tokid].head
        # print(f"\t\t\tID: {headid} token: {data[sentid][headid].form}\t{data[sentid][headid].upos}\t{data[sentid][headid].feats}")
        # print(data[sentid][headid].conll())
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


