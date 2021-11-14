# AUTOLEX #
For each linguistic question (word order, agreement, case marking) we provide a separate entry point.

### Commands
Please keep the SUD treebanks in a folder `--sud_data`, you can download them from [[here]](https://surfacesyntacticud.github.io/).
Specify the treebanks on which you to run and test the model in `decision_tree_files.txt`.
It is space separated file with entries like
`SUD_English-EWT SUD_English-LinES/en_lines-sud-test.conllu SUD_English-GUM/en_gum-sud-test.conllu`
where the model is trained on the treebank `SUD_English-EWT` and tested on the files mentioned in the line.
`--folder_name` specifies the folder in which the html files for visualizing the rules can be found.
Here are the commands which use all syntactic, lexical and semantic features.
For word order:
```
    python word_order_main.py \
    --input sud_data/ \
    --file decision_tree_files.txt \
    --lexical \
    --use_spine \
    --use_xgboost \
    --folder_name website
```
For agreement, you can specify the morphological attribute for which you want to extract the rules in `--features`.
```
    python word_order_main.py \
    --input sud_data/ \
    --file decision_tree_files.txt \
    --lexical \
    --use_spine \
    --use_xgboost \
    --features Gender+Person+Number \
    --folder_name website
```
for case marking:
```
    python case_marking_main.py \
    --input sud_data/ \
    --file decision_tree_files.txt \
    --lexical \
    --use_spine \
    --use_xgboost \
    --features Gender+Person+Number \
    --folder_name website
```
To train only on syntactic features, remove `--lexical` and `--use_spine`.
If you only want do a hyperparameter search to find which tree you want to visualize the rules from, `--no_print` to the above commands.
This will print the accuracy and number of leaves for each hyperparameter setting.
Without this, the tree performing best on the accuracy will be selected to extract rules from, for visualization.

### Semantic Features
To transform the continuous embeddings into interpretable features, run the code from [[here]](https://github.com/harsh19/SPINE).
We recommend embeddings trained over Wikipedia, which [[fasttext]](https://fasttext.cc/docs/en/pretrained-vectors.html) has released publicly for 157 languages.
Rename the embeddings after transformation to `{lang_id}_spine.txt`.
To extract interpretable features, run:
``` python getSpineFeatures.py \
    --emb es_spine.txt \
    --output es_spine_feats.txt
```
An example is shown in data/ folder for Spanish features.
