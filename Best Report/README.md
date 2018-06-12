1) Please install packages in requirements/requirements.txt (I used anaconda and libraries listed in requirements_short.txt). Check requirements/system_requirements.txt

2) download all the data (5 files: train.csv,holidays.csv,metadata.csv,weather.csv,submission_format.csv) and place them into folder: data/raw 

There are 4 notebooks in notebooks folder. To get the submission close to my best submission, I suggest You do the following steps:

3) run notebooks (use "run all"):1.0-pk-site_334_61_labeling.ipynb,2.0-pk-site_234_203_labeling.ipynb,3.0-pk-site-038-labeling.ipynb

4) check if the number of labeled anomalies (showed in the output of last cell of each notebook ) is close to mine (notebook 1.0... should give about 503, 2.0...- about 220, 3.0...- about 144). If not - run the corresponding notebook one more time. This is needed, because unfortunately I have not fixed the random seed for this challenge.
After this step, you will have three serialised dataframe objects in the folder: data/interim. These are labeled datasets for each of the building.
You will also get figures for the final report in reports/figures folder after executing notebook 1.0-pk-.... 

5) run notebook 4.0-pk-final-labeling.ipynb  . After that You will find final submission submission_3_bldngs_7111.csv in the folder submits

Comments.

The most Important here are generated figures in the folder reports/figures . They show just the same things as ones in the report sent at the end of the competition. Main steps of models creation are shown in the report, though notebooks have some more things, connected not with the methods and ideas but also with getting high score on the leaderboard (some parameters are tuned by hand for example thresholds in notebook 4.0-pk-...). 

Please check docs folder for part III. Model documentation and write-up.
