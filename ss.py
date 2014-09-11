import glob
import numpy as np
import pandas as pd
from time import time
from sklearn import decomposition as dc
from sklearn import svm 
from sklearn import grid_search as gs
from sklearn import metrics as mtrcs 
import csv

path_07 = '/home/gone/Dev/Independent/Data/SecondSpectrum/Data_07'
path_08 = '/home/gone/Dev/Independent/Data/SecondSpectrum/Data_08'
# path_09 = '/home/gone/Dev/Independent/Data/SecondSpectrum/Data_09'

files_07 = glob.glob(path_07 + "/*.csv") 
files_08 = glob.glob(path_08 + "/*.csv")  
# files_09 = glob.glob(path_09 + "/*.csv")

pbp_07 = pd.concat([pd.read_csv(f, index_col=None, header=0) for f in files_07], keys=files_07)
pbp_08 = pd.concat([pd.read_csv(f, index_col=None, header=0) for f in files_08], keys=files_07)
# pbp_09 = pd.concat([pd.read_csv(f, index_col=None, header=0) for f in files_07], keys=files_07)

#  If they appear, somehow must remove extra column headers

# Clean up data

# Filter out bad data
pbp_07 = pbp_07[pd.notnull(pbp_07['x'])]
pbp_07 = pbp_07[pd.notnull(pbp_07['y'])]

pbp_08 = pbp_08[pd.notnull(pbp_08['x'])]
pbp_08 = pbp_08[pd.notnull(pbp_08['y'])]

# Group together (ostensibly) most salient stats
simple_ind_07 = pbp_07.loc[((pbp_07['result'] == 'made') | (pbp_07['result'] =='missed')), ['x','y']]
simple_dep_07 = pbp_07.loc[((pbp_07['result'] == 'made') | (pbp_07['result'] =='missed')), ['result']].replace(['missed', 'made'], [0,1])

simple_ind_08 = pbp_08.loc[((pbp_08['result'] == 'made') | (pbp_08['result'] =='missed')), ['x','y']]
simple_dep_08 = pbp_08.loc[((pbp_08['result'] == 'made') | (pbp_08['result'] =='missed')), ['result']].replace(['missed', 'made'], [0,1])

# Normalize, project input data for speed
si07 = dc.PCA().fit_transform((simple_ind_07 - simple_ind_07.mean()) / (simple_ind_07.max() - simple_ind_07.min()))
si08 = dc.PCA().fit_transform((simple_ind_08 - simple_ind_08.mean()) / (simple_ind_08.max() - simple_ind_08.min()))

# What can we learn from the 06-07 data?
def simple_learner_07():	 
	# Train an SVM classification model
	print("Fitting the classifier to the 06-07 data")
	t0 = time()
	c = 10.0 ** np.arange(1,6,1)
	g = 10.0 ** np.arange(-3,0,.5)
	params = [{'kernel': ['rbf'], 'gamma': g, 'C': c}]
	clf = gs.GridSearchCV(svm.SVC(), params).fit(si07, simple_dep_07.values.ravel())
	print("done in %0.3fs" % (time() - t0))
	print("The best classifier found by grid search  is: ",clf.best_estimator_)
	return clf

# Does what we learned apply to the 07-08 data?

def simple_tester_08(clf):
    print("Predicting shot outcomes in the 07-08 season")
    t0 = time()
    pred_dep_08 = clf.predict(si08)
    print("done in %0.3fs" % (time() - t0))
    print(mtrcs.classification_report(simple_dep_08, pred_dep_08, target_names=target_names))
    print(mtrcs.confusion_matrix(simple_dep_08, pred_dep_08))

simple_tester_08(simple_learner_07())

# Discussion

# There are many metrics, hereafter explanatory variables or EVs, which one might expect would help predict
# the outcome of a shot. The first iteration of my prediction algorithm focuses on the shot's location, but 
# other potential EVs include:

# the number of minutes the shooter has played over the game, season, and his career;
# his height, and, if the shot was contested, the defender's height; possibly more accurately, the extension
# of the shooter when he releases the ball vs. that of his defender; 
# the proximity of the defender over the course of the shot motion;
# the direction of the shooter's gaze over the course of the shot; 
# The type of shot (jump, hook, bank)
# whether the shot was proceeded by a pass or a dribble;
# the team the shooter plays for, and is playing against

# We can extract some EVs from the data provided and but not others. For instance, we can track the number of 
# minutes each active player has played by an iterative sum (pseudocode): 
# for players in data[players]:
# 	for plays in data:
#	 	if(player.played(play)): 
#			minutes(play) = minutes(play - 1) + (data['time'][play] - data['time'][play - 1])
# 		else
#			minutes(play) = minutes(play - 1)
#	
# On the other hand, analysis using eye tracking would require much more intensive bookkeeping, since different
# shots and players may require more or less head motion or time looking at the basket. This brings up another
# consideration: many of these proposed EVs may have hierarchal interrelationships. For instance, the impact of 
# a shot's location on its outcome may differ between a fastbreak team and a 3-point shooting team. Or, how
# well actions before a shot, like a dribble or pass, predicts its outcome may differ from shooting guard to
# center or from All-Star to rookie. Such hierarchal relationships among EVs call for a multilevel model in which 
# the coefficients associated with certain predictors could be allowed to vary across different contexts. As an
# example, it would certainly be ideal to allow coefficients to vary from team to team and opponent to opponent, such that our
# model recognizes the potential different between a shot against the Raptors and the "same" shot against the Spurs.

# Contexts to consider may include:
# Shooter's team;
# Opposing team;
# Player's position;
# General skill level of player, perhaps measured by efficiency or human metric (recent All_Star appearances, 
# coaches' poll);
# Injury history (of shooter, defender)

# With all these potential EVs, we must discern which carry the most weight. Identifying which predictors explain
# the most variance in the data helps to avoid wasting computational resources unnecessarily. PCA is considered 
# the most straightforward method of projecting a data space onto a more manageable basis, but has a large drawback
# in that these bases often have no or unintelligible human meaningfulness. In order to get at a dataset's 
# "intrinsic variables", i.e. the potentially hidden true variables from which data is sampled, one may instead
# elect to use a nonlinear dimensionality reduction technique. One of particular personal interest is diffusion
# mapping, due to its computational inexpensiveness and sheer novelty.