# Logistic Regression Model (Three Optimization Algorithms)

## Usage 

Run these commands to get started :    
- `python -m venv dslr_env`     
- `source dslr_env/bin/activate`        
- `pip install -r requirements.txt`     
- `cd src && tensorboard --logdir=output/log`

Then, to launch the data visualisations : 
- `cd src`
- `python describe.py --path_csv_to_read ../data/dataset_train.csv`
- `python histogram.py --path_csv_to_read ../data/dataset_train.csv`
- `python pair_plot.py --path_csv_to_read ../data/dataset_train.csv`
- `python scatter_plot.py --path_csv_to_read ../data/dataset_train.csv`


## Provided dataset 

A Hogwarts student dataset with the following columns, splitted into two categories:

* Student information: Index, Hogwarts House, First Name, Last Name, Birthday, Best Hand

* Course scores: Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying

The main librairies for vizualisation are matplotlib (foundation) and seaborn which is built on matplotlib but it is more suited to statistical visualisations and it has great aesthetics. Plus, the syntax is less complex. Pandas has built in plotting but it is too simple for what we are being asked to produce.

## Overview of Logistic Regression Pipeline

```
Data Input → Forward Pass → Cost Calculation → Gradient Computation → Parameter Update → Repeat
```

Let's walk through each step and see how the three optimization methods handle them differently.

---

## Step 1: 🔧 **Initialization**
*Same for all optimizers*

```python
def initialize_parameter(self):
    self.W = np.zeros(self.X.shape[1])  # Weights for each feature
    self.b = 0.0                        # Bias term
```

**What happens:**
- Initialize weights W to zeros (one weight per feature)
- Initialize bias b to zero
- Set up data references (X, y, m=number of samples)

---

## Step 2: 🎯 **Forward Propagation** 
*Implementation varies by optimizer*

### **Core Forward Pass (same logic, different data)**
```python
def forward(self, X):
    Z = np.matmul(X, self.W) + self.b    # Linear combination
    A = self.sigmoid(Z)                   # Apply sigmoid activation
    return A

def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))          # Convert to probabilities [0,1]
```

### **🔀 BRANCHING: Data Selection Differs**

#### **Branch A: Gradient Descent**
```python
def gradient_descent_step(self):
    predictions = self.forward(self.X)  # ← Uses ALL data (entire dataset)
```
- **Data used**: Complete dataset (all m samples)
- **Memory**: High (processes all data at once)
- **Computation**: Most expensive per step

#### **Branch B: Stochastic Gradient Descent**
```python
def stochastic_gradient_descent_step(self):
    random_index = np.random.randint(0, self.m)
    X_sample = self.X[random_index:random_index+1]  # ← Uses 1 sample only
    y_sample = self.y[random_index:random_index+1]
    predictions = self.forward(X_sample)
```
- **Data used**: Single random sample
- **Memory**: Minimal (only 1 sample)
- **Computation**: Fastest per step

#### **Branch C: Mini-batch Gradient Descent**
```python
def mini_batch_gradient_descent_step(self):
    batch_size = params.batch_size  # e.g., 32 samples
    for batch_indices in self.get_batch_indices(batch_size):
        X_batch = self.X.iloc[batch_indices].values  # ← Uses small batch
        y_batch = self.y.iloc[batch_indices].values
        predictions_batch = self.forward(X_batch)
```
- **Data used**: Small batches (e.g., 32 samples)
- **Memory**: Moderate (batch-sized chunks)
- **Computation**: Balanced approach

---

## Step 3: 💰 **Cost Calculation**
*Same formula, different timing*

```python
def compute_cost(self, predictions, y):
    m = len(y)
    cost = np.sum((-np.log(predictions + 1e-8) * y) + 
                  (-np.log(1 - predictions + 1e-8)) * (1 - y))
    return cost / m
```

**What happens:**
- Binary cross-entropy loss calculation
- Measures how far predictions are from actual labels
- Lower cost = better model performance

### **🔀 BRANCHING: When Cost is Calculated**

#### **Branch A: Gradient Descent**
- Cost calculated **once per epoch** using full dataset
- Most accurate cost measurement
- Smooth, decreasing cost curve

#### **Branch B: SGD**
- Cost calculated using **single sample** for gradient
- But **full dataset cost** computed for monitoring
- Noisy cost curve (fluctuates up and down)

#### **Branch C: Mini-batch**
- Cost calculated per **batch** for gradients
- **Full dataset cost** computed for monitoring
- Moderately smooth cost curve

---

## Step 4: 📐 **Gradient Computation**
*Same math, different data sizes*

```python
def compute_gradient(self, X, predictions, y):
    m = len(y)
    dW = np.matmul(X.T, (predictions - y)) / m  # Gradient for weights
    db = np.sum(predictions - y) / m            # Gradient for bias
    return dW, db
```

**What happens:**
- Calculate how much to change each weight (dW)
- Calculate how much to change bias (db)
- Direction pointing toward minimum cost

### **🔀 BRANCHING: Gradient Quality**

#### **Branch A: Gradient Descent**
```
Gradient quality: ⭐⭐⭐⭐⭐ (Most accurate)
Uses all m samples → Most reliable direction
```

#### **Branch B: SGD**
```
Gradient quality: ⭐⭐ (Noisy but fast)
Uses 1 sample → Less reliable direction, more randomness
```

#### **Branch C: Mini-batch**
```
Gradient quality: ⭐⭐⭐⭐ (Good balance)
Uses batch_size samples → Good approximation of true gradient
```

---

## Step 5: ⚡ **Parameter Update**
*Same update rule, different frequencies*

```python
def update_parameters(self, dW, db):
    self.W -= self.learning_rate * dW  # Update weights
    self.b -= self.learning_rate * db  # Update bias
```

**What happens:**
- Move parameters in direction opposite to gradient
- Learning rate controls step size
- Gradually improves model performance

### **🔀 BRANCHING: Update Frequency**

#### **Branch A: Gradient Descent**
```
Updates per epoch: 1
Update timing: After processing entire dataset
Step size: Large, stable steps
```

#### **Branch B: SGD**
```
Updates per epoch: m (number of samples)
Update timing: After each single sample
Step size: Small, frequent, noisy steps
```

#### **Branch C: Mini-batch**
```
Updates per epoch: m/batch_size (e.g., if m=1000, batch=32 → 31 updates)
Update timing: After each batch
Step size: Medium steps, good stability
```

---

## Step 6: 🔄 **Training Loop**
*Different convergence patterns*

```python
def fit(self, X, y, callback, house_name):
    for epoch in range(self.epochs):
        all_predictions = self.perform_optimization_step()  # ← Branches here
        cost = self.compute_cost(all_predictions, self.y)
        self.cost_history.append(cost)
        self.log_progress(epoch, cost, callback, house_name)
```

### **🔀 BRANCHING: Convergence Behavior**

#### **Branch A: Gradient Descent**
```
Convergence: Smooth, predictable descent
Cost curve: ╲╲╲╲╲╲╲ (steady downward)
Time per epoch: Slow (processes all data)
Memory usage: High
Best for: Small datasets, when you want stability
```

#### **Branch B: SGD**
```
Convergence: Fast but noisy
Cost curve: ╲╱╲╱╲╱╲ (zigzag downward)
Time per epoch: Fast (processes one sample at a time)
Memory usage: Low
Best for: Large datasets, when you want speed
```

#### **Branch C: Mini-batch**
```
Convergence: Balanced speed and stability
Cost curve: ╲⩕╲⩕╲⩕╲ (slightly bumpy downward)
Time per epoch: Medium
Memory usage: Medium
Best for: Most practical situations (default choice)
```

---

## 🎯 **Complete Training Flow Comparison**

### **Gradient Descent Flow:**
```
Epoch 1: Process ALL 1000 samples → Compute gradients → Update once
Epoch 2: Process ALL 1000 samples → Compute gradients → Update once
...
```

### **SGD Flow:**
```
Epoch 1: Sample 1 → Update → Sample 2 → Update → ... → Sample 1000 → Update
Epoch 2: Sample 847 → Update → Sample 23 → Update → ... (random order)
...
```

### **Mini-batch Flow:**
```
Epoch 1: Batch 1 (32 samples) → Update → Batch 2 → Update → ... → Batch 32 → Update
Epoch 2: Shuffle data → Batch 1 → Update → Batch 2 → Update → ...
...
```

## 🎓 **Key Takeaways**

1. **Same Goal**: All three methods try to minimize the same cost function
2. **Different Paths**: They take different approaches to get there
3. **Trade-offs**: Speed vs. Stability vs. Memory usage
4. **Use Cases**: 
   - Small data → Gradient Descent
   - Large data → SGD
   - Most cases → Mini-batch (best of both worlds)

The core logistic regression math stays the same - only the data processing strategy changes!
    

## Quick overview of the Linear Regression and Logistic Regression methods : 

📈 Linear Regression   

Goal	: Predict a continuous value (e.g., price, temperature)   
Output	: A real number (e.g., 4.2, 101.7)   
Model function	: A linear function: `y = θ₀ + θ₁x`   
Error metric	: Mean Squared Error (MSE), MAE, etc.   
Example	: Predicting a house price based on its size   

📊 Logistic Regression   

Goal	: Predict a probability of class membership (e.g., yes/no, 0/1)   
Output	: A probability between 0 and 1   
Model function	: A sigmoid function: `σ(z) = 1 / (1 + e^(-z)), where z = θ₀ + θ₁x`  
Error metric	: Log-loss (cross-entropy loss)  
Example	: Predicting whether a student is admitted (1) or not (0) based on their score  
LogReg is then the best approach to handle a multiclass binary classification task like this Hogwarts Hat problem !   


## Data Visualisation
 
We start by calculating the statistics of our preprocessed dataset which can be found in `output/describe/statistics.csv`. 
Some useful statistical concepts:

✅ MEAN: The average — the sum of all values divided by the number of values.   
✅ STD or Standard Deviation: A measure of how spread out the data is around the mean. A higher standard deviation means the values are more dispersed from the average.     
✅ QUARTILES: Three values that split sorted data into four parts, each with an equal number of observations Quartiles & Quantiles | Calculation, Definition & Interpretation.    Specifically:   
- Q1 (First quartile): The 25th percentile, meaning that 25% of the data falls below the first quartile Quartiles & Quantiles | Calculation, Definition & Interpretation
- Q2 (Second quartile/Median): The 50th percentile, meaning that 50% of the data falls below the second quartile Quartiles & Quantiles | Calculation, Definition & Interpretation
- Q3 (Third quartile): The 75th percentile, meaning that 75% of the data falls below the third quartile Quartiles & Quantiles | Calculation, Definition & Interpretation
✅ MIN/MAX: The smallest/largest value in the dataset.   
✅ COUNT NaN: The number of missing or null values in each column.   
✅ PERCENT NaN: The percentage of missing values relative to the total dataset size.   
✅ VARIANCE: The average of the squared differences from the mean. It measures variability in the data (standard deviation squared).   
✅ RANGE: The difference between the maximum and minimum values (Max - Min).   

<details>
<summary><h5 style="display: inline; margin: 0;" >CSV File tip</h5></summary>   

Download the extension "Rainbow CSV" and at the bottom of the IDE there's a clickable addon **align** so you can visualise them better. Don't forget to remove it and switch it back to **shrink** as it will invalidate any further parsing (it adds spaces). Shrink vs Align : 

<p align="center">
  <img src="./assets/shrink.png" width="400" height="200"/>
  <img src="./assets/align.png" width="400" height="200"/>
</p>

</details>  

<details>
<summary><h4 style="display: inline; margin: 0;"> 1. Which Hogwarts course has a homogeneous score distribution between all four houses? (Histogram)</h4></summary>

A histogram is a graphical representation that organizes data into continuous intervals or "bins," displaying the frequency or count of observations within each bin. A great ressource [here](https://www.coursera.org/fr-FR/articles/what-is-a-histogram).

A "homogeneous score distribution between all four houses" means the score distributions are similar across all houses. This is actually the *opposite* of what we want for effective classification. For good classification, we want features where each house shows distinct patterns.

<details><summary>SOLUTION 1 : Simple histogram</summary>

Firstly, we simply generated a histogram using seaborn. We get the following which displays quite obvisouly courses in which students' grades are homogeneous vs non-homogenous : 

<p align="center">
  <img src="./assets/homogeneous.png" width = "500" height="400" />
  <img src="./assets/nonhomogeneous.png" width = "500" height="400" />
</p>

Tried something by preprocessing Birthday and Best Hand but it is too homogeneous to be relevant. 

<p align="center">
  <img src="./assets/agehand.png" width = "500" height="400" />
</p>

</details>

<details><summary>SOLUTION 2 : Use F-RATIO, a homogeneity metric </summary>

- Between-Group Variance: Measures how different the house means are from each other for a given course. Higher values indicate greater differences between houses.
- Within-Group Variance: Measures how much scores vary within each house. Lower values indicate more consistency within houses.
- F-ratio: between-variance ÷ average within-variance 

The course with the lowest F-ratio would be considered the most homogeneous, as this indicates minimal differences between houses relative to the variation within houses.

<details><summary> An example that explains F-Ratio </summary>
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

</details> 


<details>
<summary><h4 style="display: inline; margin: 0;"> 2. What are the two features that are similar ? (Scatter plots)</h4></summary>
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

</details> 

<details>
<summary><h4 style="display: inline; margin: 0;">  3. From this visualization, what features are you going to use for your logistic regression? (Pair plot) </h4></summary> 

We can see here that interesting features are Herbology for example which allows us to really distinguish students from different houses. 

<p align="center">
  <img src="./assets/pair_plot.png" />
</p>
</details> 