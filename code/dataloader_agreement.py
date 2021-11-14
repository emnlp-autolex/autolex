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
        self.feature_dictionary = defaultdict(lambda : 0)
        self.feature_map = {}
        self.labels = defaultdict(lambda : 0)
        self.triple_freq = defaultdict(lambda: 0)
        self.feature_map_pos = {}
        self.feature_freq = defaultdict(lambda :0)
        self.required_features = []#['Tense', 'Animacy', 'Aspect', 'Mood', 'Definite', 'Voice', 'Poss','PronType', 'VerbForm']
        self.required_relations = ['obj', 'mod', 'det', 'mod', 'subj', 'vocative', 'aux', 'compound', 'conj', 'flat', 'appos']  # ,
        self.required_relations = []
        self.remove_features = ['Gender', 'Person', 'Number']
        random.seed(args.seed)

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

    def readData(self, inputFiles, genreparseddata, wikidata, prop, vocab_file, spine_word_vectors, spine_features, spine_dim):
        for num, inputFile in enumerate(inputFiles):
            if not inputFile:
                continue
            genreparseddata_ = genreparseddata[num]
            self.lang_full = inputFile.strip().split('/')[-2].split('-')[0][3:]
            f = inputFile.strip()
            data = pyconll.load_from_file(f"{f}")

            is_test = False
            if "test" in inputFile:
                is_test = True
            index = 0

            for sentence in data:
                text = sentence.text
                id2index = sentence._ids_to_indexes
                token_data, genre_token_data = None, None
                if self.args.use_wikid and genreparseddata_ and text in genreparseddata_:
                    genre_token_data = genreparseddata_[text]

                # Add the head-dependents
                dep_data_token = defaultdict(list)
                for token_num, token in enumerate(sentence):
                    dep_data_token[token.head].append(token.id)

                for token_num, token in enumerate(sentence):
                    token_id = token.id
                    if "-" in token_id or "." in token_id and not self.isValid(token.deprel):
                        continue
                    relation = token.deprel
                    pos = token.upos
                    feats = token.feats
                    lemma = token.lemma

                    if prop not in feats:
                        continue
                    label = self.getFeatureValue(prop, feats)

                    if token.head and token.head != "0" and pos and relation:
                        relation = relation.lower()
                        head_pos = sentence[token.head].upos
                        head_feats = sentence[token.head].feats
                        headrelation = sentence[token.head].deprel
                        headhead = sentence[token.head].head
                        head_lemma = sentence[token.head].lemma

                        self.feature_dictionary[label] += 1

                        if prop  in head_feats:
                            headlabel = self.getFeatureValue(prop, head_feats)
                            feature = f'deppos_{pos}'
                            if not is_test:
                                self.feature_freq[feature] += 1

                            feature = f'headpos_{head_pos}'
                            if not is_test:
                                self.feature_freq[feature] += 1

                            if 'unk' not in relation:
                                feature = f'deprel_{relation}'
                                if not is_test:
                                    self.feature_freq[feature] += 1

                            if not self.args.only_triples: #adding dependents of the dep

                                for feat in feats: #self.required_features:
                                    if prop in feat or prop in self.remove_features:#skip the propery being modeled
                                        continue
                                    feature = f'depfeat_{feat}_'
                                    value = self.getFeatureValue(feat, feats)
                                    feature += f'{value}'
                                    if not is_test:
                                        self.feature_freq[feature] += 1


                                if headrelation:
                                    headrelation = headrelation.lower()
                                    if headrelation != 'root' and headrelation != 'punct' and "unk" not in headrelation:
                                        feature = f'headrelrel_{headrelation.lower()}'
                                        if not is_test:
                                            self.feature_freq[feature] += 1

                                for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
                                    if prop in feat or prop in self.remove_features:
                                        continue
                                    feature = f'headfeat_{head_pos}_{feat}_'
                                    value = self.getFeatureValue(feat, head_feats)
                                    feature += f'{value}'
                                    if not is_test:
                                        self.feature_freq[feature] += 1

                                    feature = f'headfeat_{feat}_{value}'
                                    if not is_test:
                                        self.feature_freq[feature] += 1

                                # Adding lemma of the head's head
                                if self.args.lexical:


                                    lemma = self.isValidLemma(lemma, pos)
                                    if lemma:
                                        feature = f'lemma_{lemma}'
                                        if not is_test:
                                            self.feature_freq[feature] += 1

                                    head_lemma = self.isValidLemma(head_lemma, head_pos)
                                    if head_lemma:
                                        feature = f'headlemma_{head_lemma}'
                                        if not is_test:
                                            self.feature_freq[feature] += 1

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
                                                    self.feature_freq[feature] += 1

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
                                                self.feature_freq[feature] += 1

                                    depdeppos = sentence[d].upos
                                    feature = f'depheadpos_{depdeppos}'
                                    if not is_test:
                                        self.feature_freq[feature] += 1

                                if headhead and headhead != '0':
                                    headhead_feats = sentence[headhead].feats
                                    if prop in headhead_feats:
                                        headhead_value =   self.getFeatureValue(prop, headhead_feats)
                                        if headlabel == headhead_value:
                                            feature = f'headmatch_True'
                                            if not is_test:
                                                self.feature_freq[feature] += 1

                            if self.args.use_wikid and genre_token_data:
                                if token_id in genre_token_data:
                                    qids_all = genre_token_data[token_id]
                                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                                    for feature in features:
                                        if not is_test:
                                            feature = 'wiki_' + feature
                                            self.feature_freq[feature] += 1

                                if token.head in genre_token_data:
                                    qids_all = genre_token_data[token.head]
                                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                                    for feature in features:
                                        feature = 'wiki_head_' + feature
                                        if not is_test:
                                            self.feature_freq[feature] += 1

                            if self.args.use_spine:
                                vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                                feature_names = spine_features[vector == 1]  # Get active features
                                for feature in feature_names:
                                    feature = f'spine_{feature}'
                                    if not is_test:
                                        self.feature_freq[feature] += 1

                                if token.head and token.head != '0':
                                    vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim, sentence[token.head].lemma)
                                    feature_names = spine_features[vector == 1]  # Get active features
                                    for feature in feature_names:
                                        feature = f'spinehead_{feature}'
                                        if not is_test:
                                            self.feature_freq[feature] += 1

                            if label == headlabel: #If agreement between the head-dep
                                agreelabel = 1
                            else:
                                agreelabel = 0
                            self.labels[agreelabel] += 1

                            index += 1

        self.pos_id2tag = {v: k for k, v in self.pos_dictionary.items()}
        self.feature_map_pos_id2tag = {}
        with open(vocab_file, 'w') as fout:

            self.feature_map_pos = {}
            for feature, freq in self.feature_freq.items():
                if freq < 50:
                    continue
                self.feature_map_pos[feature] = len(self.feature_map_pos)
            self.feature_map_pos_id2tag = {v:k for k,v in  self.feature_map_pos.items()}
            fout.write(f'POS:All\n')
            for k,v in self.feature_map_pos.items():
                fout.write(f'{v}\t{k}\n')
            fout.write('\n')
            fout.write(f'Data:All\tLabels\t')
            label_values = []
            for label, v in self.labels.items():
                label_values.append(str(label) + "," + str(v))
            fout.write(";".join(label_values) + "\n")

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
            if lemma not in spine_word_vectors:
                spine_word_vectors[lemma] = vector
        elif lemma in spine_word_vectors:
            vector = spine_word_vectors[lemma]
        return np.array(vector)

    def isValid(self,relation):
        if not relation:
            return False
        found = False

        relation = relation.lower()
        if len(self.required_relations) > 0:  # Restrict analysis to relations
            for rel in self.required_relations:
                if rel in relation:
                    found = True
        else:
            found = True
        return found

    def addFeature(self, feature, feature_array):
        feature_id = self.feature_map_pos.get(feature, -1)
        if feature_id >= 0:
            feature_array[feature_id] = 1

    def getAssignmentFeatures(self, inputFile,
                                  genreparseddata,
                                  wikidata, prop,
                              spine_word_vectors, spine_features, spine_dim, filename):
        f = inputFile.strip()
        data = pyconll.load_from_file(f"{f}")
        all_features = []
        columns = []
        output_labels = defaultdict(lambda : 0)
        num_of_tokens_syntax = 0
        num_of_tokens_wikigenre = 0
        num_of_tokens_spine = 0
        num_of_tokens_lexical = 0

        index = [i for i in range(len(data))]
        #Get column names
        for i in range(len(self.feature_map_pos)):
            feature_name = self.feature_map_pos_id2tag[i]
            columns.append(feature_name)
        columns.append('label')
        columns.append('sent_num')
        columns.append('token_num')
        num = 0
        for sentence_num in index:
            sentence = data[sentence_num]
            text = sentence.text
            token_data, genre_token_data = None, None
            if self.args.use_wikid and genreparseddata and text in genreparseddata:
                genre_token_data = genreparseddata[text]

            # Add the head-dependents
            dep_data_token = defaultdict(list)
            for token_num, token in enumerate(sentence):
                dep_data_token[token.head].append(token.id)

            for token_num, token in enumerate(sentence):
                token_id = token.id
                if "-" in token_id or "." in token_id and not self.isValid(token.deprel):
                    continue
                relation = token.deprel.lower()
                pos = token.upos
                feats = token.feats
                lemma = token.lemma

                if prop not in feats:
                    continue
                label = self.getFeatureValue(prop, feats)
                feature_array = np.zeros((len(self.feature_map_pos),), dtype=int)

                if token.head and token.head != "0" and pos and relation:
                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    headrelation = sentence[token.head].deprel
                    headhead = sentence[token.head].head
                    head_lemma = sentence[token.head].lemma


                    if prop in head_feats:
                        headlabel = self.getFeatureValue(prop, head_feats)

                        feature = f'deppos_{pos}'
                        self.addFeature(feature, feature_array)

                        feature = f'headpos_{head_pos}'
                        self.addFeature(feature, feature_array)

                        feature = f'deprel_{relation}'
                        self.addFeature(feature, feature_array)

                        if not self.args.only_triples:  # adding dependents of the dep
                            for feat in feats:  # self.required_features:
                                if prop in feat:  # skip the propery being modeled
                                    continue
                                feature = f'depfeat_{feat}_'
                                value = self.getFeatureValue(feat, feats)
                                feature += f'{value}'
                                self.addFeature(feature, feature_array)

                            if headrelation and headrelation != 'root' and headrelation != 'punct':
                                feature = f'headrelrel_{headrelation.lower()}'
                                self.addFeature(feature, feature_array)

                            for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
                                if prop in feat:
                                    continue
                                feature = f'headfeat_{head_pos}_{feat}_'
                                value = self.getFeatureValue(feat, head_feats)
                                feature += f'{value}'
                                self.addFeature(feature, feature_array)

                                feature = f'headfeat_{feat}_{value}'
                                self.addFeature(feature, feature_array)

                            if headhead and headhead != '0':
                                headhead_feats = sentence[headhead].feats
                                headheadhead = sentence[headhead].head
                                headhead_value = None
                                if prop in headhead_feats:
                                    headhead_value = self.getFeatureValue(prop, headhead_feats)
                                    if headlabel == headhead_value:
                                        feature = f'headmatch_True'
                                        self.addFeature(feature, feature_array)

                            # Adding lemma of the head's head
                            if self.args.lexical:
                                lemma = self.isValidLemma(lemma, pos)
                                if lemma:
                                    feature = f'lemma_{lemma}'
                                    self.addFeature(feature, feature_array)

                                head_lemma = self.isValidLemma(head_lemma, head_pos)
                                if head_lemma:
                                    feature = f'headlemma_{head_lemma}'
                                    self.addFeature(feature, feature_array)

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
                                            self.addFeature(feature, feature_array)

                            # get other dep tokens of the head
                            dep = dep_data_token.get(token.head, [])
                            for d in dep:
                                if d == token.id:
                                    continue
                                depdeprelation = sentence[d].deprel
                                if depdeprelation and depdeprelation != 'punct':
                                    feature = f'depheadrel_{depdeprelation}'
                                    self.addFeature(feature, feature_array)

                                depdeppos = sentence[d].upos
                                feature = f'depheadpos_{depdeppos}'
                                self.addFeature(feature, feature_array)


                        if self.args.use_wikid and genre_token_data:
                            isWiki = False
                            if token_id in genre_token_data:
                                qids_all = genre_token_data[token_id]
                                features = self.getWikiDataGenreParse(qids_all, wikidata)
                                for feature in features:
                                    feature = 'wiki_' + feature
                                    self.addFeature(feature, feature_array)
                                    isWiki = True
                                if isWiki:
                                    num_of_tokens_wikigenre += 1
                            isWiki = False
                            if token.head in genre_token_data:
                                qids_all = genre_token_data[token.head]
                                features = self.getWikiDataGenreParse(qids_all, wikidata)
                                for feature in features:
                                    feature = 'wiki_head_' + feature
                                    self.addFeature(feature, feature_array)
                                    isWiki = True

                                if isWiki:
                                    num_of_tokens_wikigenre += 1

                        if self.args.use_spine:
                            isspine = False
                            vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                            feature_names = spine_features[vector == 1]  # Get active features
                            for feature in feature_names:
                                feature = f'spine_{feature}'
                                self.addFeature(feature, feature_array)
                                isspine = True

                            if isspine:
                                num_of_tokens_spine +=  1

                            isspine = False
                            if token.head and token.head != '0':
                                vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim,
                                                               sentence[token.head].lemma)
                                feature_names = spine_features[vector == 1]  # Get active features
                                for feature in feature_names:
                                    feature = f'spinehead_{feature}'
                                    self.addFeature(feature, feature_array)
                                    isspine = True
                                if isspine:
                                    num_of_tokens_spine += 1

                        if label == headlabel:  # If agreement between the head-dep
                            agreelabel = 1
                        else:
                            agreelabel = 0

                        one_feature = np.concatenate(
                            (feature_array, int(agreelabel), sentence_num, token.id), axis=None)
                        assert len(one_feature) == len(columns)
                        all_features.append(one_feature)
                        output_labels[agreelabel] += 1
                        num += 1

        print(
            f'Syntax: {num_of_tokens_syntax}  Wikigenre: {num_of_tokens_wikigenre}, Lexical: {num_of_tokens_lexical}, Spine:{num_of_tokens_spine} columns: {len(columns)}')
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
            req_head_head = False
            for feature in active:
                info = feature.split("_")
                if len(info) > 2:
                    if feature.startswith("agree") or feature.startswith('headrelrel'):
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


