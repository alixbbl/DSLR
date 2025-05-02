
## Rappel :

Run these commands to get started : 

`python3 -m venv dslr_env`   
`source dslr_env/bin/activate`      
`pip install -r requirements.txt`    


## Our dataset 

A Hogwarts student dataset with the following columns, that we've splitted into two categories:

* Student information: Index, Hogwarts House, First Name, Last Name, Birthday, Best Hand

* Course scores: Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying

<details> <summary> Librairies used</summary>
The main librairies for vizualisation are matplotlib (foundation) and seaborn which is built on matplotlib but it is more suited to statistical visualisations and it has great aesthetics. Plus, the syntax is less complex. Pandas has built in plotting but is too simple for what we are being asked to produce.
</details>

# Pre-process our dataset (describe.py)

Since we can eventually only handle numeric data in our visualisatoins, let's transform the dataset to keep relevant informations but have them as numeric informations :
- turn birthdays into a new column age
- turn best hand into a new column hand binary with 0 for left-handed and 1 for right-handed

<details> <summary> CSV File tip</summary>
Download the extension "Rainbow CSV" and at the bottom of the IDE there's a clickable addon **align** so you can visualise them better. Don't forget to remove it and switch it back to **shrink** as it will invalidate any further parsing (it adds spaces). Shrink vs Align : 

<p align="center">
  <img src="./assets/shrink.png" width="400" height="400"/>
  <img src="./assets/align.png" width="400" height="400"/>
</p>

</details>

Eventually, we calculate the statistics of our preprocessed dataset which can be found in `output/describe/statistics.csv`. Quelques notions de statistiques utiles :

✅ MEAN : 
La moyenne, la somme des valeurs divisee par le nombre de ces valeurs.

✅ STD ou Ecart-type :
C'est la mesure de dispersion des donnees par rapport a la moyenne de ces donnees.
Plus un STD est grand, plus les donnees son eclatees par rapport a leur moyenne.

✅ Quartiles :
Ils divisent un ensemble de donnees en 4 parties egales : jusqu'a 25%, 50%, 75% et 100%. 
Cela permet d'avoir une idee de la distribution des valeurs du dataset.


## Which Hogwarts course has a homogeneous score distribution between all four houses? (Histogram)

A histogram is a graphical representation that organizes data into continuous intervals or "bins," displaying the frequency or count of observations within each bin. A great ressource [here](https://www.coursera.org/fr-FR/articles/what-is-a-histogram).

A "homogeneous score distribution between all four houses" means the score distributions are similar across all houses. This is actually the *opposite* of what we want for effective classification. For good classification, we want features where each house shows distinct patterns.

<details> <summary>SOLUTION 1 : Simple histogram</summary>

Firstly, we simply generated a histogram using seaborn. We get the following which displays quite obvisouly courses in which students' grades are homogeneous vs non-homogenous : 

<p align="center">
  <img src="./assets/homogeneous.png" width = "500" height="400" />
  <img src="./assets/nonhomogeneous.png" width = "500" height="400" />
</p>

We had to switch the dataset from having one row per student to get one row per grade in a 

</details>

<details> <summary>SOLUTION 2 : Use F-RATIO, a homogeneity metric </summary>

- Between-Group Variance: Measures how different the house means are from each other for a given course. Higher values indicate greater differences between houses.
- Within-Group Variance: Measures how much scores vary within each house. Lower values indicate more consistency within houses.
- F-ratio: between-variance ÷ average within-variance 

The course with the lowest F-ratio would be considered the most homogeneous, as this indicates minimal differences between houses relative to the variation within houses.

<details> <summary> An example that explains F-Ratio </summary>
Scenario 1:

Course A: House means are [70, 72, 73, 71]
Between-group variance = 1.5
Every student in each house gets nearly identical scores (within-group variance ≈ 0)

Scenario 2:

Course B: House means are also [70, 72, 73, 71]
Between-group variance = 1.5 (same as Course A)
Scores within each house vary wildly from 40-100 (within-group variance = 225)

Both courses have identical between-group variance, but they tell completely different stories:

In Course A, the houses truly perform differently (the small differences are meaningful)
In Course B, the houses aren't meaningfully different because the within-house variation dwarfs the between-house differences

The F-ratio as a Solution
The F-ratio (between-variance ÷ within-variance) solves this by contextualizing the between-group differences:

Course A: F-ratio = 1.5 ÷ ~0 = very high → truly heterogeneous
Course B: F-ratio = 1.5 ÷ 225 = 0.0067 → actually homogeneous
</details>

Low F-ratio: Indicates similar means across houses with similar internal variations (homogeneous)
High F-ratio: Indicates significant differences between houses (heterogeneous)

<p align="center">
  <img src="./assets/metrics.png" width = "800" height="400" />
</p>

</details>


## What are the two features that are similar ? (Scatter plots)
A scatter plot allows you to visualize relationships between two variables. Its name comes from the graph's design—it looks like a collection of dots scattered across an x- and y-axis. A great ressource [here](https://www.coursera.org/articles/what-is-a-scatter-plot).

The correlation coefficient measures the statistical relationship between two variables. We use corr() on our dataset (only on the numeric values) to find which subjects are closest. The corr() function in pandas computes pairwise correlation between columns, returning a correlation matrix that shows how each variable relates to every other variable. Range of Values:

+1.0: Perfect positive correlation (as one variable increases, the other increases proportionally)
0: No correlation (variables are independent)
-1.0: Perfect negative correlation (as one variable increases, the other decreases proportionally)

By default, it uses Pearson correlation, which measures linear relationships between variables. Tried the other methods but the most "definite" one for our values is Pearson. We notice in some courses, one house is very separate from the rest. We see a perfect negative correlation : 

<p align="center">
  <img src="./assets/correlation_matrix.png" />
</p>

<p align="center">
  <img src="./assets/scatter_similar.png" />
</p>

## From this visualization, what features are you going to use for your logistic regression? (Pair plot) 

# Logistic Regression OnevsAll

## Train

Inherently, machine learning models are binary classifiers. Logistic regression is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not.

The One-Versus-All (OvR) method decomposes a multi-class problem into multiple binary classification tasks, where each class is trained against all others using logistic regression.

<p align="center">
  <img src="./assets/onevsall.png" width="500" height="400" />
</p>

**The issue at hand : to which Hogwart house does the student belong ?**

So in our case, there are 4 different "classes" which corresponds to the 4 different Hogwarts houses : Gryffindor, Hufflepuff, Ravenclaw, Slytherin. So N = 4, and we'll need to train 4 independent classifiers.


## Predict

So we got trained our function to get the parameters (weights) which will give us the lowest cost (distance between truth and predicted). If we now predict on unseen (but labelled) data, let's see how accurate our model is. 

Common metrics : 
- Accuracy: Accuracy provides the proportion of correctly classified instances.
- Precision: Precision focuses on the accuracy of positive predictions.
- Recall (Sensitivity or True Positive Rate): Recall measures the proportion of correctly predicted positive instances among all actual positive instances.
- F1 Score: F1 score is the harmonic mean of precision and recall.
