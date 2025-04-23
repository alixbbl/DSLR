
## Rappel :

Run these commands to get started : 

`python3 -m venv dslr_env`   
`source dslr_env/bin/activate`      
`pip install -r requirements.txt`    

## Quelques notions de statistiques utiles :

✅ MEAN : 
La moyenne, la somme des valeurs divisee par le nombre de ces valeurs.

✅ STD ou Ecart-type :
C'est la mesure de dispersion des donnees par rapport a la moyenne de ces donnees.
Plus un STD est grand, plus les donnees son eclatees par rapport a leur moyenne.

✅ Quartiles :
Ils divisent un ensemble de donnees en 4 parties egales : jusqu'a 25%, 50%, 75% et 100%. 
Cela permet d'avoir une idee de la distribution des valeurs du dataset.

# Data Visualisation 

The main librairies for vizualisation are matplotlib (foundation) and seaborn which is built on matplotlib but it is more suited to statistical visualisations and it has great aesthetics. Plus, the syntax is less complex. These are the only ones we use at this point in the project. Pandas has built in plotting but is too simple for what we are being asked to produce. Seaborn used to have a method called distplot() which is now deprecated. The way to create a histogram in seaborn is with histplot(). 

## Which Hogwarts course has a homogeneous score distribution between all four houses? (Histogram)
A histogram is a graphical representation that organizes data into continuous intervals or "bins," displaying the frequency or count of observations within each bin. A great ressource [here](https://www.coursera.org/fr-FR/articles/what-is-a-histogram).

A "homogeneous score distribution" in this case means a distribution whereby of each Hogwart house will be similar (ie the bins will overlap). Below is an example of homogeneous vs non-homogeneous score distribution in two courses at Hogwart.

<p align="center">
  <img src="./assets/homogenous.png" width="500" height="400" />
  <img src="./assets/homogenous_non.png" width="500" height="400"/>
</p>


## What are the two features that are similar ? (Scatter plots)
A scatter plot allows you to visualize relationships between two variables. Its name comes from the graph's design—it looks like a collection of dots scattered across an x- and y-axis.  A great ressource [here](https://www.coursera.org/articles/what-is-a-scatter-plot).

We're asked to answer the super vague question above. So we'd say that to answer "What are the two features that are similar?" using a scatter plot, we're essentially trying to determine which two _features_ (columns) are most correlated — either positively (both increase together) or negatively (one increases while the other decreases). This means the scatter plot should show a clear linear or patterned relationship between those two features. 

What this will mean in terms of our data is that these will be the subjects in which students who perform well in one subject also tend to perform well in the other (and vice versa for low scores). If Charms and Herbology have a correlation of, say, 0.75, it suggests that students who score high in Charms often also score high in Herbology, and those who do poorly in one also tend to do poorly in the other. This could mean : similar skills are involved, there are eaching overlaps or even that the same group of students are studying together for these subjects.

The steps we're taking : 

- We choose numerical columns only, so we ignore categorical ones (like House, Name, Birthday, Best Hand).
- We use corr() to get the correlation between the courses
- We scatter plot correlated courses (i.e the ones above a certain threshold)

Generally correlations above 0.7 are considered strong.

For example : Students who score high in Astronomy consistently score low in Defense Against the Dark Arts and Students who score low in Astronomy consistently score high in Defense Against the Dark Arts.

<p align="center">
  <img src="./assets/correlated_neg.png" width="500" height="400" />
</p>


Another example : Students who perform well in Herbology generally also perform well in Charms and Students who struggle with Herbology tend to also struggle with Charms.

<p align="center">
  <img src="./assets/correlated.png" width="500" height="400" />
</p>

## pair plot 





