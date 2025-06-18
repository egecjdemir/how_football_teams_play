# How Football Teams Play

final_version folder is the final version of the project. old_version can be ignored.

In order to reproduce the project results, following steps must be followed:
1. download the events and matches data from this link: https://github.com/koenvo/wyscout-soccer-match-event-dataset
2. put extracted json files with the same directory with the notebooks.
3. run the notebooks in create_datasets folder.
4. run the notebooks in train folder.

the notebooks starts with "dec_train", produces the 3 scores for all 7 methods, for k values in [2,10] range, in their respective game phases.

further analysis like loss plot and win percentage are conducted in final_training_with_dec_in_poss notebook, for only in possesion data for simplicity.
