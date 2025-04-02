Predicting Car Sales Price
Project Description
This project, Predicting Car Sales Price, aims to develop a predictive model for estimating the selling price of cars based on historical sales data. By analyzing various attributes such as make, model, year, condition, odometer reading, and more, the project provides actionable insights for stakeholders in the automotive industry. The model assists dealerships in optimizing pricing strategies, auction houses in setting realistic reserve prices, and buyers in making informed purchasing decisions. This initiative enhances market efficiency and ensures fair valuations.

Key Findings & Results
Exploratory Data Analysis (EDA):

Cars with lower mileage and better condition tend to have higher selling prices.

The Market Monitor Retail (MMR) value strongly correlates with selling prices, validating its reliability for pricing benchmarks.

Most cars were sold within the price range of $1,000 to $3,000.

Predictive Models:

Linear Regression Model: Achieved an R² value of 0.9705 and Mean Squared Error (MSE) of 2,693,294.37. This model effectively captures the relationship between car attributes and selling price.

Random Forest Model: Outperformed linear regression slightly with an R² value of 0.9733 and MSE of 2,436,633.75. While less interpretable than linear regression, it demonstrated higher predictive accuracy.

Tools & Technologies Used
Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

Modeling Techniques: Linear Regression and Random Forest

Data Visualization Tools: Histograms, Scatter Plots, Correlation Matrices

How to Run the Code
Setup Environment:

Install Python (version 3.8 or later).

Install required libraries using:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
Download Data:

Ensure the dataset is placed in the project directory.

Execute Scripts:

Run the preprocessing script to clean and prepare the data:

bash
python data_preprocessing.py
Execute the exploratory data analysis script:

bash
python eda.py
Train models using the modeling script:

bash
python predictive_modeling.py
View Results:

Outputs include cleaned datasets, visualizations (saved as images), and model performance metrics displayed in the console.