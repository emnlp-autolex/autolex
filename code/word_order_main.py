import  os, pyconll
import  dataloader_word_order as dataloader
import argparse
import numpy as np

np.random.seed(1)
import sklearn
import sklearn.tree
from collections import defaultdict
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from io import StringIO
import utils
import pydotplus
from sklearn.tree import  export_text
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb


DTREE_DEPTH = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
DTREE_IMPURITY = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
XGBOOST_GAMMA = [0.005, 0.01, 1, 2, 5, 10]

def printTreeFromXGBoost(model, rel, folder_name, train_data, train_df , lang, tree_features, obs_label, relation_map, setting=""):
    best_model = model.best_estimator_
    rules_df = best_model.get_booster().trees_to_dataframe()
    rules_df_t0 = rules_df.loc[rules_df['Tree'] == 0]
    # Iterate the tree to get information about the tree features
    topnodes, tree_dictionary, leafnodes, leafedges, edge_mapping = utils.iterateTreesFromXGBoost(rules_df_t0, args.task, rel,
                                                                                       relation_map, tree_features)

    #Assign statistical threshold to each leaf
    leaf_examples, leaf_sent_examples = utils.FixLabelTreeFromXGBoost(rules_df_t0, tree_dictionary, leafnodes, label_list, data_loader.model_data_case[rel],
                                                 rel, train_df, train_data,
                                                 task=args.task)

    tree_dictionary, leafmap = utils.collateTreeFromXGBoost(tree_dictionary, topnodes, leafnodes, leafedges)


    print("Num of leaves : ", len(leafmap))
    if args.no_print:
        return tree_dictionary, leafmap, {}, {}, {}


    retainNA = True
    # Get the features in breadth first order of the class which deviates from the dominant order
    important_features, cols = utils.getImportantFeatures(tree_dictionary, leafmap, '', retainNA)
    if len(important_features) == 0:  # If there is no sig other label
        return tree_dictionary, leafmap, {}, {}, {}

    leaf_examples_features = {}

    #Iterate each leaf of the tree to populate the table
    total_num_examples = 0
    num_examples_label = defaultdict(lambda: 0)
    num_leaf_examples = {}
    leaf_values_per_columns, columns_per_leaf, update_spine_features  = {}, {}, {}

    with open(f'{folder_name}/{lang}_{rel}_rules.txt', 'w') as fleaf:
        for leaf_num in range(len(leafmap)):
            leaf_examples_features[leaf_num] = {}
            columns_per_leaf[leaf_num] = defaultdict(list)
            leaf_index = leafmap[leaf_num]
            leaf_label = tree_dictionary[leaf_index]['class']
            leaf_examples_features[leaf_num]['leaf_label'] = leaf_label

            # Get examples which agree and disagree with the leaf-label
            examples = leaf_examples[leaf_index]
            sent_examples = leaf_sent_examples[leaf_index]


            agree, disagree, total_agree, total_disagree, spine_features = utils.getExamples(sent_examples,
                                                                             tree_dictionary[leaf_index],
                                                                             train_data, isTrain=True)

            for keyspine, valuespines in spine_features.items():
                key = keyspine.split("_")[0]
                sorted_spine = sorted(valuespines.items(), key=lambda kv:kv[1], reverse=True)[:10]
                values = [v for (v,_) in sorted_spine]
                update_spine_features[keyspine] = f'{key}_{",".join(values)}'
            leaf_examples_features[leaf_num]['agree'] = agree
            leaf_examples_features[leaf_num]['disagree'] = disagree

            fleaf.write(f'agree:{total_agree}, disagree: {total_disagree}, total: {total_agree + total_disagree}\n')

            row = f'<tr>'

            active_all = tree_dictionary[leaf_index]['active'] #features which are active for this leaf
            non_active_all = tree_dictionary[leaf_index]['non_active'] #features which are not active for this leaf
            top = tree_dictionary[leaf_index].get('top', -1)

            while top > 0:  # Not root
                active_all += tree_dictionary[top]['active']
                non_active_all += tree_dictionary[top]['non_active']
                top = tree_dictionary[top]['top']

            active, non_active = [],[]
            for (feat, _) in active_all:
                feat = update_spine_features.get(feat, feat)
                active.append(feat)
            for (feat, _) in non_active_all:
                feat = update_spine_features.get(feat, feat)
                non_active.append(feat)

            leaf_examples_features[leaf_num]['active'] = active
            leaf_examples_features[leaf_num]['non_active'] = non_active

            fleaf.write(leaf_label + "\n")
            fleaf.write('active: ' + ",".join(active) + "\n")
            fleaf.write('non_active: ' + ",".join(non_active) + "\n")
            fleaf.write("\n")
            # populating only rules which deviate from the dominant label as observed from the training data
            if not retainNA and leaf_label == 'NA':
                continue
            for feat in important_features:
                feat = update_spine_features.get(feat, feat)
                if feat in active:

                    columns_per_leaf[leaf_num]['Y'].append(feat)
                    row += f'<td style=\"text-align:center\"> Y </td>'
                elif feat in non_active:

                    columns_per_leaf[leaf_num]['N'].append(feat)
                    row += f'<td style=\"text-align:center\"> N </td>'
                else:

                    columns_per_leaf[leaf_num]['-'].append(feat)
                    row += f'<td style=\"text-align:center\"> - </td>'
            row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>\n'
            leaf_examples_features[leaf_num]['row'] = row

            #get the number of examples predicted for each label
            for _, num_example_per_leaf in sent_examples.items():
                num_examples_label[leaf_label] += len(num_example_per_leaf)
                total_num_examples += len(num_example_per_leaf)

            num_leaf_examples[leaf_num] = total_agree + total_disagree

        #Get the dominant order for printing
        print_leaf = []
        max_label = None
        max_value = 0
        prob = []
        for leaf_label, num_examples in num_examples_label.items():
            print_leaf.append(f'{leaf_label}: {num_examples}, {num_examples / total_num_examples}')
            prob.append(num_examples / total_num_examples)
            if num_examples > max_value:
                max_value = num_examples
                max_label = leaf_label

        print_leaf.append(f'Word order: {max_label}')
        fleaf.write(f'Total Examples: {total_num_examples}\n\n')
        H = entropy(prob, base=2)
        print_leaf.append(f'Entropy in training data: {H}')
        print("\n".join(print_leaf))
        print()

    return tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf

def xgboostModel(rel, x,y, x_test, y_test, depth,cv):
    criterion = ['gini', 'entropy']
    parameters = {'criterion': criterion, 'max_depth': depth, 'min_child_weight': [20]}

    xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1, subsample=0.8,
                      colsample_bytree=0.8, objective='multi:softprob', num_class=len(label_list), silent=True, nthread=args.n_thread,
                      scale_pos_weight=majority / minority, seed=1001)

    model = GridSearchCV(xgb_model, parameters, cv=cv, n_jobs=args.n_jobs)
    model.fit(x, y)

    best_model = model.best_estimator_

    dtrainPredictions = best_model.predict(x)
    x_test = x_test.to_numpy()
    dtestPredictions = best_model.predict(x_test)

    train_acc = accuracy_score(y, dtrainPredictions) * 100.0
    test_acc = accuracy_score(y_test, dtestPredictions) * 100.0
    print(
        "Model: %s, Lang: %s, Train Accuracy : %.4g, Test Accuracy : %.4g, Baseline Test Accuracy: %.4g, Baseline Label: %s" % (
            rel, lang_full, train_acc, test_acc, baseline_acc, baseline_label))
    print("Hyperparameter tuning: ", model.best_params_)
    return model

def train(train_data, train_df,
          train_features, train_label,
          dev_features, dev_label,
          test_features, test_label, label_list,
          rel, outp, folder_name,
          baseline_acc, baseline_label,
          lang, test_datasets, tree_features ,
          relation_map,best_depths, minority, majority):

    x_train, y_train, x_test, y_test = train_features, train_label, test_features, test_label

    tree_features = np.array(tree_features)

    if dev_features is not None and not dev_features.empty:
        x_dev, y_dev = dev_features, dev_label
        x = np.concatenate([x_train, x_dev])
        y = np.concatenate([y_train, y_dev])
        test_fold = np.concatenate([
            # The training data.
            np.full(x_train.shape[0], -1, dtype=np.int8),
            # The development data.
            np.zeros(x_dev.shape[0], dtype=np.int8)
        ])
        cv = sklearn.model_selection.PredefinedSplit(test_fold)
    else:
        x, y = x_train, y_train
        cv = 5
        if args.use_xgboost:
            x = x.to_numpy()



    #Print acc/leaves for diff settings
    if args.no_print:
        for depth in DTREE_DEPTH:  #:
            setting = f'-depth-{depth}'
            model = xgboostModel(rel, x, y, x_test, y_test, [depth], cv)
            best_model = model.best_estimator_
            best_model.get_booster().dump_model(f'{folder_name}/{rel}/xgb_model-{setting}.txt', with_stats=True)
            tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf = printTreeFromXGBoost(model, rel, folder_name, train_data, train_df, lang, tree_features, baseline_label,
                                 relation_map,
                                 setting)
        print()
    else:
        if rel in best_depths and best_depths[rel] != -1:
            model = xgboostModel(rel, x, y, x_test, y_test, [int(best_depths[rel])], cv)
            best_model = model.best_estimator_
            best_model.get_booster().dump_model(f'{folder_name}/{rel}/xgb_model.txt', with_stats=True)
            tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf = printTreeFromXGBoost(model, rel, folder_name, train_data, train_df, lang, tree_features, baseline_label,
                                 relation_map
                                 )
        else:
            model = xgboostModel(rel, x, y, x_test, y_test, DTREE_DEPTH, cv)
            best_model = model.best_estimator_
            best_model.get_booster().dump_model(f'{folder_name}/{rel}/xgb_model.txt', with_stats=True)
            tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf = printTreeFromXGBoost(model, rel, folder_name, train_data, train_df, lang, tree_features, baseline_label,
                                 relation_map,
                                 )


    best_model = model.best_estimator_
    outp.write(f'<h4> <a href=\"{rel}/{rel}.html\">{rel}</a></h4>\n')
    retainNA = args.retainNA

    with open(f"{folder_name}/{rel}/{rel}.html", 'w') as output:
        HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")
        output.write(HEADER + '\n')
        output.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
                   f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
                   f'<li class="nav"><a href="../../../about.html\">About Us</a></li></ul>\n')
        output.write(f"<br><li><a href=\"../WordOrder.html\">Back to {language_fullname} page</a></li>\n")
        model_info = rel.split("-")
        output.write(
            f'<h3> Order of <b>{model_info[0]}s</b> with respect to the syntactic head <b>{model_info[1]}</b> </h3>\n')
        output.write(f'<div id="\label\"> <h3> The dominant order in the corpus is <b>{baseline_label}</b></div>\n')

        dominant_label = baseline_label
        active_features_per_nondomleaf, columns_with_activeinactive_features, column_with_leafinfo, columns_active_indomleaves, columns_in_innondomleaves = [], {}, {}, set(), set()

        if not args.no_print:
            # Sort leaves by number of examples, and remove the col from first general leaf
            sorted_leaves = sorted(num_leaf_examples.items(), key=lambda kv: kv[1], reverse=True)
            first_leaf, columns_to_remove, columns_to_retain = True, set(), set()


            for (leaf_num, _) in sorted_leaves:
                leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                if not retainNA and leaf_label == 'NA':
                    continue

                if first_leaf and leaf_label == dominant_label:  # This is the leaf which says the general word-order
                    leaf_examples_features[leaf_num]['cols'] = ['generally word order is ']
                    leaf_examples_features[leaf_num]['default'] = True
                    dominant_label = leaf_label
                    first_leaf = False

                active_columns_in_leaf = columns_per_leaf[leaf_num].get('Y', [])
                nonactive_columns_in_leaf = columns_per_leaf[leaf_num].get('N', [])
                missing_columns_in_leaf = columns_per_leaf[leaf_num].get('-', [])

                if leaf_label != dominant_label and len(
                        active_columns_in_leaf) > 0:  # There is at least one active feature in the leaf
                    active_features_per_nondomleaf += active_columns_in_leaf + nonactive_columns_in_leaf

                all_cols = set(active_columns_in_leaf + nonactive_columns_in_leaf + missing_columns_in_leaf)
                for c in all_cols:
                    if c in active_features_per_nondomleaf:
                        if leaf_label != dominant_label: #feature is present for a non-dominant leaf
                            columns_in_innondomleaves.add(c)

                        columns_to_retain.add(c)
                        if c not in columns_with_activeinactive_features:
                            columns_with_activeinactive_features[c] = defaultdict(lambda: 0)

                        if c in active_columns_in_leaf:
                            columns_with_activeinactive_features[c]['Y'] += 1
                            if leaf_label == dominant_label:
                                columns_active_indomleaves.add(c)

                        elif c in nonactive_columns_in_leaf or c in missing_columns_in_leaf:
                            columns_with_activeinactive_features[c]['N'] += 1

            for c, leafdata in columns_with_activeinactive_features.items():
                leaves_with_active_column = leafdata.get('Y', 0)
                if leaves_with_active_column == 0 and c not in columns_in_innondomleaves:
                    columns_to_retain.remove(c)

            for c in columns_active_indomleaves:  # for col active in dominant leaves, check if its active (Y,N) in non-dom leaves
                to_keep = False
                for (leaf_num, _) in sorted_leaves:
                    leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                    if not retainNA and leaf_label == 'NA':
                        continue

                    if leaf_label != dominant_label:
                        active_columns_in_leaf = columns_per_leaf[leaf_num].get('Y', [])
                        nonactive_columns_in_leaf = columns_per_leaf[leaf_num].get('N', [])

                        if c in active_columns_in_leaf or c in nonactive_columns_in_leaf:
                            to_keep = True

                    if to_keep:
                        break

                if not to_keep:
                    columns_to_retain.remove(c)

            # Create the table of features
            columns_to_retain = list(columns_to_retain)
            if len(columns_to_retain) == 0 and not args.no_print: #all NA leaves:
                return

            output.write(
                f'<table><col><colgroup span=\"{len(columns_to_retain)}\"></colgroup>'
                f'<tr><th colspan=\"{len(columns_to_retain)}	scope=\"colgroup\" \" style=\"text-align:center\">Rules</th>'
                f'<th rowspan=\"3\" style=\"text-align:center\">Label</th>'
                f'<th rowspan=\"3\" style=\"text-align:center\">Examples</th>'
                f'<th rowspan=\"3\" style=\"text-align:center\">Test Examples</th></tr><tr>\n')

            col_names = ["" for _ in range(len(columns_to_retain))]
            cols = utils.getColsToCombine(columns_to_retain)
            for combined in cols:
                header, subheader = utils.getHeader(combined, columns_to_retain, rel, args.task, relation_map)
                output.write(
                    f'<th colspan=\"{len(combined)}	scope=\"colgroup\" \" style=\"text-align:center;width:130px\">{header}</th>\n')
                for col, header in zip(combined, subheader):
                    col_names[col] = header

            output.write('</tr><tr>')
            for col, feat in enumerate(col_names):
                output.write(f'<th scope=\"col\" style=\"text-align:center\"> {feat} </th>\n')
            output.write('</tr>')

            first_leaf, leaves_covered = True, set()
            for (leaf_num, _) in sorted_leaves:
                leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                if not retainNA and leaf_label == 'NA':
                    continue
                if first_leaf:
                    leaf_examples_file = f'{folder_name}/{rel}/Leaf-{leaf_num}.html'
                    active, non_active = leaf_examples_features[leaf_num]['active'], leaf_examples_features[leaf_num][
                        'non_active']
                    agree, disagree = leaf_examples_features[leaf_num]['agree'], leaf_examples_features[leaf_num][
                        'disagree']

                    if leaf_examples_features[leaf_num].get('default', False):
                        row = f'<tr> <th colspan=\"{len(columns_to_retain)}	scope=\"colgroup\" \" style=\"text-align:center;width:130px\">Generally the word order is </th>\n'
                        row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>'
                        utils.populateLeaf(train_data, leaf_examples_file, leaf_label, agree, disagree,
                                           active, non_active, first_leaf, rel, args.task, lang_full, relation_map,
                                           columns_to_retain, ORIG_HEADER, HEADER,
                                           FOOTER)
                        cols = ['default']
                        leaf_examples_features[leaf_num]['valid'] = True

                    else:
                        yes_cols, no_cols, missin_cols = columns_per_leaf[leaf_num].get('Y', []), columns_per_leaf[
                            leaf_num].get('N', []), columns_per_leaf[leaf_num].get('-', [])
                        row = '<tr>'
                        num_features_in_leaf, cols = 0, []
                        for column in columns_to_retain:  # Iterate all columns
                            if column in yes_cols:
                                row += f'<td style=\"text-align:center\"> Y </td>'
                                num_features_in_leaf += 1
                                cols.append(f'Y_{column}')
                            elif column in no_cols:
                                row += f'<td style=\"text-align:center\"> N </td>'
                                num_features_in_leaf += 1
                                cols.append(f'N_{column}')
                            else:
                                row += f'<td style=\"text-align:center\"> - </td>'
                                cols.append(f'-_{column}')

                        row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>'

                        if num_features_in_leaf > 0:  # the row shares at least one 'Y' feature with the non-dominant label
                            leaf_examples_features[leaf_num]['valid'] = True
                            utils.populateLeaf(train_data, leaf_examples_file, leaf_label, agree, disagree,
                                               active, non_active, False, rel, args.task, lang_full, relation_map,
                                               columns_to_retain, ORIG_HEADER, HEADER,
                                               FOOTER)
                        else:
                            leaf_examples_features[leaf_num]['valid'] = False
                    leaf_examples_features[leaf_num]['row'] = row
                    leaf_examples_features[leaf_num]['dispcols'] = cols
                    first_leaf = False
                    cols.sort()
                    leaf_cols = f'{leaf_label}{";".join(list(cols))}'
                    leaves_covered.add(leaf_cols)
                else:
                    yes_cols, no_cols, missin_cols = columns_per_leaf[leaf_num].get('Y', []),  columns_per_leaf[leaf_num].get('N', []), columns_per_leaf[leaf_num].get('-', [])
                    row = '<tr>'

                    num_features_in_leaf, cols, yes_cols_number, no_cols_number, missing_cols_number = 0, [], 0, 0, 0
                    for column in columns_to_retain:  # Iterate all columns
                        if column in yes_cols:
                            row += f'<td style=\"text-align:center\"> Y </td>'
                            num_features_in_leaf += 1
                            yes_cols_number += 1
                            cols.append(f'Y_{column}')
                        elif column in no_cols:
                            row += f'<td style=\"text-align:center\"> N </td>'
                            num_features_in_leaf += 1
                            no_cols_number += 1
                            cols.append(f'N_{column}')
                        else:
                            row += f'<td style=\"text-align:center\"> - </td>'
                            cols.append(f'-_{column}')
                            missing_cols_number += 1

                    cols.sort()
                    leaf_cols = f'{leaf_label}{";".join(list(cols))}'
                    print(leaf_num, leaf_cols, yes_cols_number, leaves_covered)
                    if leaf_cols in leaves_covered or yes_cols_number == 0:  # remove leaves which have all features 'N' or "-"
                        leaf_examples_features[leaf_num]['valid'] = False
                        continue

                    row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>\n'
                    leaf_examples_features[leaf_num]['row'] = row
                    leaf_examples_features[leaf_num]['dispcols'] = set(cols)

                    leaf_examples_features[leaf_num]['valid'] = True
                    leaf_examples_file = f'{folder_name}/{rel}/Leaf-{leaf_num}.html'
                    active, non_active = leaf_examples_features[leaf_num]['active'], leaf_examples_features[leaf_num]['non_active']
                    agree, disagree = leaf_examples_features[leaf_num]['agree'], leaf_examples_features[leaf_num][
                        'disagree']
                    utils.populateLeaf(train_data, leaf_examples_file, leaf_label, agree, disagree,
                                       active, non_active, first_leaf, rel, args.task, lang_full, relation_map, columns_to_retain, ORIG_HEADER, HEADER, FOOTER)

                    leaves_covered.add(leaf_cols)

        for (new_test_features, new_test_label, new_test_lang, new_test_df, new_test_path, debug) in test_datasets:
            y_baseline = label_encoder.transform([baseline_label] * len(new_test_df))
            new_baseline_acc = accuracy_score(new_test_label, y_baseline) * 100.0
            if args.use_xgboost:
                new_test_features = new_test_features.to_numpy()
            new_dtestPredictions = best_model.predict(new_test_features)
            new_test_acc = accuracy_score(new_test_label, new_dtestPredictions) * 100.0

            print("Test Path: %s" % (new_test_path))
            print("Model: %s, Lang: %s, Test Accuracy : %.4g, Baseline Accuracy: %.4g, Baseline Label: %s" % (rel, new_test_lang, new_test_acc, new_baseline_acc, baseline_label))
            if args.no_print:
                continue
            with open(debug, 'w') as fout:
                for i in range(len(new_test_features)):
                    datapoint = new_test_df.iloc[i]
                    label = new_test_label[i]
                    pred = new_dtestPredictions[i]
                    if label != pred:
                        fout.write(str(datapoint['sent_num']) + "," + str(datapoint['token_num']) + "," + str(datapoint['label']) + '\n')
            new_test_data = pyconll.load_from_file(f"{new_test_path}")

            total_num_examples = 0
            num_examples_label = defaultdict(lambda : 0)

            first_leaf = True
            for (leaf_num, _) in sorted_leaves:
                leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                if not retainNA and leaf_label == 'NA':
                    continue

                valid = leaf_examples_features[leaf_num]['valid']
                if not valid:
                    continue

                leaf_index = leafmap[leaf_num]
                examples = {}
                sent_examples = defaultdict(list)
                utils.getExamplesPerLeaf(leaf_index, tree_dictionary, rel, new_test_df, new_test_data, args.task,
                                         examples, sent_examples)

                agree, disagree, total_agree, total_disagree, _ = utils.getExamples(sent_examples,
                                                                                    tree_dictionary[leaf_index],
                                                                                    new_test_data)

                # Populate the leaf with examples
                leaf_examples = f'{folder_name}/{rel}/Leaf-{leaf_num}-{new_test_lang}.html'
                active, non_active = leaf_examples_features[leaf_num]['active'], leaf_examples_features[leaf_num][
                    'non_active']
                print(f'{new_test_lang},{leaf_num},{leaf_label},{total_agree},{total_disagree}')
                if len(agree) > 0 or len(disagree) > 0:
                    row = leaf_examples_features[leaf_num]['row']
                    row += f'@@@<a href=\"Leaf-{leaf_num}-{new_test_lang}.html\"> {new_test_lang} </a><br> '
                    leaf_examples_features[leaf_num]['row'] = row

                if first_leaf:
                    utils.populateLeaf(new_test_data, leaf_examples, leaf_label, agree, disagree,
                                       active, non_active, first_leaf, rel, args.task,
                                       lang_full, relation_map, columns_to_retain, ORIG_HEADER, HEADER, FOOTER)
                    first_leaf = False
                else:
                    utils.populateLeaf(new_test_data, leaf_examples, leaf_label, agree, disagree,
                                       active, non_active, first_leaf, rel, args.task,
                                       lang_full, relation_map, columns_to_retain, ORIG_HEADER, HEADER, FOOTER)

                # get the number of examples predicted for each label
                for _, num_example_per_leaf in sent_examples.items():
                    num_examples_label[leaf_label] += len(num_example_per_leaf)
                    total_num_examples += len(num_example_per_leaf)

            print_leaf = []
            max_label = None
            max_value = 0

            prob = []
            for leaf_label, num_examples in num_examples_label.items():
                print_leaf.append(f'{leaf_label}: {num_examples}, {num_examples/total_num_examples}')
                prob.append(num_examples)
                if num_examples > max_value:
                    max_value = num_examples
                    max_label = leaf_label
            print_leaf.append(f'Test Word order: {max_label}')

            H = entropy(prob, base=2)
            print_leaf.append(f'Test Entropy: {H}')
            print("\n".join(print_leaf))
            print()

        if args.no_print:
            return
        #sort leaves by examples
        sorted_leaves = sorted(num_leaf_examples.items(), key=lambda kv:kv[1], reverse=True)
        for (leaf_num, _) in sorted_leaves:
            leaf_label = leaf_examples_features[leaf_num]['leaf_label']
            if not retainNA and leaf_label == 'NA':
                continue

            valid = leaf_examples_features[leaf_num]['valid']
            print(leaf_num, leaf_label, valid)
            if not valid:
                continue

            row = leaf_examples_features[leaf_num]['row']
            test_rows = row.split("@@@")
            row = test_rows[0] + '<td>' + " ".join(test_rows[1:]) + '</td>'
            row = row.replace('NA', 'Cannot decide')
            output.write(f'{row}</tr>\n')
        output.write('</table>\n')
        output.write("</ul><br><br><br>\n" + FOOTER + "\n")
    print()

def createPage(foldername):
    filename = foldername + "/" + "WordOrder.html"
    with open(filename, 'w') as outp:
        HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
        outp.write(HEADER + '\n')
        outp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
                   f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
                   f'<li class="nav"><a href=\"../../about.html\">About Us</a></li></ul>')
        outp.write(f'<h1>Rules for the following WordOrder relations:</h1>')

def filterData(data_loader):
    """ Only train models for those relations/POS which have at least 50 examples for every label"""
    classes = {}
    for model, count in data_loader.model_data.items():
        if count < 100: #Overall number of examples for that model is less than 100
            continue

        labels = data_loader.model_data_case[model]
        if len(labels) == 1:
            print(f'{model} always has  {labels[0]}  syntactic head')
            continue

        to_retain_labels = []
        max_label = 0
        max_dir = ''
        for label, value in labels.items():
            if value < 50:
                continue
            to_retain_labels.append((label, value))

        if len(to_retain_labels) == 1:
            print(f'{model} majority of the is  {to_retain_labels[0]}  syntactic head')
            continue
        if len(to_retain_labels) == 0:
            continue
        classes[model] = to_retain_labels
        #print( model, to_retain_labels)
    #print()
    return classes

def createHomePage():
    with open(f"{folder_name}/index.html", 'a') as op:
        lang_id = lang.split("_")[0]
        language_name = language_fullname.split("-")[0]
        treebank_name = language_fullname.split("-")[1]
        op.write(f'<tr><td>{lang_id}</td> '
                 f'<td>{language_name}</td> '
                 f'<td> {treebank_name} </td>'
                 f' <td> <li> <a href=\"{lang}/Agreement/Agreement.html\">Agreement</a></li>'
                 f'<li> <a href=\"{lang}/WordOrder/WordOrder.html\">WordOrder</a></li>'
                 f'<li> <a href=\"{lang}/CaseMarking/CaseMarking.html\">CaseMarking</a></li> </td></tr>\n')

    try:
        os.mkdir(f"{folder_name}/{lang}/")
    except OSError:
        i = 0
    try:
        os.mkdir(f"{folder_name}/{lang}/WordOrder")
    except OSError:
        i = 0


    with open(f"{folder_name}/{lang}/index_wordorder.html", 'w') as outp:
        HEADER = ORIG_HEADER.replace("main.css", "../main.css")
        outp.write(HEADER + "\n")
        outp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../index.html\">Home</a>'
                   f'</li><li class="nav"><a href=\"../introduction.html\">Usage</a></li>'
                   f'<li class="nav"><a href=\"../about.html\">About Us</a></li></ul>')
        outp.write(f"<br><a href=\"../index.html\">Back to language list</a><br>")
        outp.write(f"<h1> {language_fullname} </h1> <br>\n")
        outp.write(
            f'<h3> We  present  a  framework that automatically creates a first-pass specification of word order rules for various WALs features (82A, 83A, 85A, 87A, 89A and 90A.) from raw text for the language in question.</h3>')
        outp.write(
            "<h3> We parsed the <a href=\"https://universaldependencies.org\">Universal Dependencies</a> (UD) data in order to extract these rules.</h3>\n")
        outp.write(f"<br><strong>{language_fullname}</strong> exhibits the following assignment:<br><ul>")
        outp.write(
            f"<li>  <a href=\"Agreement/Agreement.html\">Agreement</a></li>\n")
        outp.write(
            f"<li>  <a href=\"CaseMarking/CaseMarking.html\">Case Assignment</a></li>\n")
        outp.write(
            f"<li>  <a href=\"WordOrder/WordOrder.html\">Word Order</a></li>\n")
        outp.write("</ul><br><br><br>\n" + FOOTER + "\n")
    filename = f"{folder_name}/{lang}/WordOrder/"
    createPage(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/sud-data//')
    parser.add_argument("--file", type=str, default="./decision_tree_files.txt")
    parser.add_argument("--relation_map", type=str, default="./relation_map")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folder_name", type=str, default='./syntax_all/',
                        help="Folder to hold the rules, need to add header.html, footer.html always")

    parser.add_argument("--features", type=str, default="WordOrder")
    parser.add_argument("--task", type=str, default="wordorder")

    parser.add_argument("--sud", action="store_true", default=True, help="Enable to read from SUD treebanks")
    parser.add_argument("--auto", action="store_true", default=False,
                        help="Enable to read from automatically parsed data")
    parser.add_argument("--noise", action="store_true", default=False,
                        help="Enable to read from automatically parsed data")

    parser.add_argument("--prune", action="store_true", default=True)
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--retainNA", action="store_true", default=True)

    parser.add_argument("--no_print", action="store_true", default=False,
                        help="To not print the website pages")
    parser.add_argument("--lang", type=str, default=None)

    # Different features to experiment with
    parser.add_argument("--only_triples", action="store_true", default=False,
                        help="Only use relaton, head-pos, dep-pos, disable to use other features")

    parser.add_argument("--use_wikid", action="store_true", default=False,
                        help="Add features from WikiData, requires args.wiki_path and args.wikidata")
    parser.add_argument("--wiki_path", type=str,
                        default="babelnet_outputs/outputs/",
                        help="Contains entity identification for nominals")
    parser.add_argument("--wikidata", type=str,
                        default="./wikidata_processed.txt", help="Contains the WikiData property for each Qfeature")

    parser.add_argument("--lexical", action="store_true", default=False,
                        help="Add lexicalized features for head and dep")

    parser.add_argument("--use_spine", action="store_true", default=False,
                        help="Add features from spine embeddings, read from args.spine_outputs path.")
    parser.add_argument("--spine_outputs", type=str, default="./spine_outputs/")
    parser.add_argument("--continuous", action="store_true", default=True,
                        help="Add features from spine embeddings, read from args.spine_outputs path.")

    parser.add_argument("--best_depth", type=int, nargs='+', default=[-1, -1, -1, -1, -1], help="Set an integer betwen [3,10] to print the website,"
                                                                  "order ['subj-verb', 'obj-verb' ,'adj-noun', 'num-noun', 'noun-adp']")
    parser.add_argument("--skip_models", type=str, nargs='+', default=[])
    parser.add_argument("--use_xgboost", action="store_true", default=True)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_thread", type=int,default=1)
    args = parser.parse_args()

    folder_name = f'{args.folder_name}'
    with open(f"{folder_name}/header.html") as inp:
        ORIG_HEADER = inp.readlines()
    ORIG_HEADER = ''.join(ORIG_HEADER)

    with open(f"{folder_name}/footer.html") as inp:
        FOOTER = inp.readlines()
    FOOTER = ''.join(FOOTER)

    #populate the best-depth to print the rules for treebank mentioned in args.file
    #If all values are -1 then it will search for the best performing depth by accuracy
    best_depths = {'subject-verb': args.best_depth[0],
                   'object-verb': args.best_depth[1],
                   'adjective-noun': args.best_depth[2],
                   'numeral-noun': args.best_depth[3],
                   'noun-adposition': args.best_depth[4],
    }

    with open(args.file, "r") as inp:
        files = []
        test_files = defaultdict(set)
        for file in inp.readlines():
            if file.startswith("#"):
                continue
            file_info = file.strip().split()
            if args.lang and args.lang not in file_info:
                continue
            files.append(f'{args.input}/{file_info[0]}')
            if len(file_info) > 1: #there are test files mentioned
                for test_file in file_info[1:]:
                    test_file = f'{args.input}/{test_file}'
                    test_files[f'{args.input}/{file_info[0]}'].add(test_file.lstrip().rstrip())

    d = {}
    relation_map = {}
    with open(args.relation_map, "r") as inp:
        for line in inp.readlines():
            info = line.strip().split(";")
            key = info[0]
            value = info[1]
            relation_map[key] = value
            if '@x' in key:
                relation_map[key.split("@x")[0]] = value

    index_file = f"{folder_name}/index.html"
    # Read train/dev/test from UD/SUD/auto-parsed

    for fnum, treebank in enumerate(files):
        train_path, dev_path, test_path, lang = utils.getTreebankPaths(treebank.strip(), args)
        test_files[treebank].add(test_path)
        if train_path is None:
            print(f'Skipping the treebank as no training data!')
            continue

        language_fullname = "_".join(os.path.basename(treebank).split("_")[1:])
        lang_full = lang
        lang_id = lang.split("_")[0]
        if args.auto:
            lang = f'{lang}_auto'
        elif args.noise:
            lang = f'{lang}_noise'


        #creating the website home page; it requires to have all html files present in that args.folder_name
        createHomePage()

        #Get wiki-data features as obtained from GENRE entity-linking tool
        genre_train_data, genre_dev_data, genre_test_data, wikiData = utils.getWikiFeatures(args, lang)

        #Get spine embedding
        spine_word_vectors, spine_features, spine_dim = None, [], 0
        if args.use_spine:
            spine_word_vectors, spine_features, spine_dim = utils.loadSpine(args.spine_outputs, lang_id, args.continuous)

        # Decision Tree code
        f = train_path.strip()
        train_data = pyconll.load_from_file(f"{f}")
        data_loader = dataloader.DataLoader(args, relation_map)

        # Creating the vocabulary
        input_dir = f"{folder_name}/{lang}"
        vocab_file = input_dir + f'/vocab.txt'
        inputFiles = [train_path, dev_path, test_path]
        data_loader.readData(inputFiles,[genre_train_data, genre_dev_data, genre_test_data],
                                 wikiData,
                                 vocab_file,
                             spine_word_vectors,
                             spine_features,
                                spine_dim)
        # Get the model info i.e. which models do we have to train e.g. subject-verb, object-verb and so on.
        modelsData = filterData(data_loader)
        filename = f"{folder_name}/{lang}/WordOrder/"
        baseline_acc = 0.0
        with open(filename + '/' + 'WordOrder.html', 'a') as outp:
            outp.write(f"<br><a href=\"../../index.html\">Back to language list</a><br>")

            for model, to_retain_labels in modelsData.items():
                if args.skip_models and model in args.skip_models:
                    continue
                try: #Create the model dir if not already present
                    os.mkdir(f"{folder_name}/{lang}/WordOrder/{model}/")
                except OSError:
                    i = 0

                train_file = f'{filename}/{model}/train.feats' #Stores the features
                dev_file = f'{filename}/{model}/dev.feats'
                test_file = f'{filename}/{model}/test.feats'
                columns_file = f'{filename}/{model}/column.feats'
                freq_file = f'{filename}/{model}/feats.freq' #Stores the freq of the features in the training data

                labels = []
                for (label, _) in to_retain_labels:
                    labels.append(label)

                if not (os.path.exists(train_file) and os.path.exists(test_file)):
                    # creating the features for training
                    train_features, columns, output_labels = data_loader.getFeatures(train_path,  model,
                                                                     labels,  genre_train_data, wikiData,
                                                                     spine_word_vectors, spine_features, spine_dim,
                                                                     train_file)
                    train_df = pd.DataFrame(train_features, columns=columns)
                    try:
                        updated_columns, valid = utils.removeFeatures(train_df)
                    except Exception as e:
                        continue

                    if not valid:
                        continue

                    train_df = train_df[updated_columns]
                    train_df.to_csv(train_file, mode='w', header=True, index=False)

                    dev_df = None
                    if dev_path:
                        dev_features, _, output_labels = data_loader.getFeatures(dev_path, model, labels, genre_dev_data,
                                                                   wikiData, spine_word_vectors, spine_features, spine_dim,
                                                                   dev_file)
                        dev_df = pd.DataFrame(dev_features, columns=columns)
                        dev_df = dev_df[updated_columns]
                        dev_df.to_csv(dev_file, mode='w', header=True, index=False)


                    test_features, _, output_labels = data_loader.getFeatures(test_path, model, labels,
                                                               genre_test_data, wikiData,
                                                               spine_word_vectors, spine_features, spine_dim,
                                                               test_file)
                    test_df = pd.DataFrame(test_features, columns=columns)
                    test_df = test_df[updated_columns]
                    test_df.to_csv(test_file, mode='w', header=True, index=False)

                    with open(freq_file, 'w') as fout:
                        sorted_feats = sorted(data_loader.feature_freq[model].items(), key=lambda kv: kv[1], reverse=False)
                        for (feature, freq) in sorted_feats:
                            fout.write(feature + "," + str(freq) + '\n')

                #Reload the df again, as some issue with first loading, print(f'Reading train/dev/test from {lang}/{pos}')
                print(f'Loading train/dev/test...')
                train_df, dev_df, test_df, columns = utils.reloadData(train_file, dev_file, test_file)

                if test_df.empty:
                    continue
                with open(freq_file, 'r') as fin:
                    data_loader.feature_freq[model] = {}
                    for line in fin.readlines():
                        info = line.strip().split(",")
                        feature = info[0]
                        value = int(info[-1])
                        data_loader.feature_freq[model][feature] = value

                train_features, train_labels, \
                dev_features, dev_labels, \
                test_features, test_labels, \
                baseline_label, \
                id2label, label_list, label_encoder, minority, majority = utils.getModelTrainingData(train_df, dev_df, test_df)

                #Get baseline accuracy when we assign the most dominant order to all test examples (most dominant order == WALS)
                y_baseline = label_encoder.transform([baseline_label] * len(test_df))
                baseline_acc = accuracy_score(test_labels, y_baseline) * 100.0
                tree_features = train_features.columns.to_list()

                test_df_info = test_df[['label', 'sent_num', 'token_num']]
                test_datasets = []

                test_paths = list(test_files[treebank])
                print(f'Feature Extraction for Test Data')
                for new_test_path in test_paths:
                    new_lang = new_test_path.strip().split('/')[-1].split("-")[0]
                    new_test_file = f'{filename}/{model}/test.{new_lang}.feats'
                    if not os.path.exists(new_test_file):
                        _, _, new_genre_test_data, _ = utils.getWikiFeatures(args, new_lang, test=True)
                        new_test_features, columns, _ = data_loader.getFeatures(new_test_path, model, labels,
                                                                          new_genre_test_data, wikiData,
                                                                         spine_word_vectors, spine_features, spine_dim, new_test_file)
                        new_test_df = pd.DataFrame(new_test_features, columns=columns)
                        new_test_df = new_test_df[np.concatenate((tree_features, ['label', 'sent_num', 'token_num']))]
                        new_test_df.to_csv(new_test_file, mode='w', header=True, index=False)

                    new_test_df = pd.read_csv(new_test_file, sep=',')

                    new_test_features, new_test_label = new_test_df.drop(
                        columns=['label', 'sent_num', 'token_num'],
                        axis=1), label_encoder.transform(
                        new_test_df[["label"]])
                    debug = f'{filename}/debug.{new_lang}.{model}.txt'

                    test_datasets.append((new_test_features, new_test_label, new_lang, new_test_df, new_test_path, debug))
                print('Training Started...')

                try:
                    train(train_data, train_df, train_features, train_labels,
                          dev_features, dev_labels,
                          test_features, test_labels,
                          label_list, model,
                          outp, filename, baseline_acc, baseline_label,
                          lang_full, test_datasets, tree_features ,
                          relation_map, best_depths, minority, majority)
                except Exception as e:
                    print(f'ERROR: Skipping {lang_full} - {model}')
                    continue

            outp.write("</ul><br><br><br>\n" + FOOTER + "\n")

