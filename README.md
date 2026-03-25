# Food-Demand-Forecasting
Smart Food Demand Forecasting
# About the Project

Food demand in meal delivery services changes frequently due to pricing, promotions, and customer behavior. Predicting this demand accurately is important because food is perishable — overestimation leads to wastage, and underestimation leads to shortages.

This project builds a machine learning-based system that predicts future food demand using historical data from multiple fulfillment centers. The system is designed to assist in better inventory planning and supply chain decisions.

# How the System Works

The system follows a simple pipeline:

1️⃣ Data Preparation

Multiple datasets such as meal details, fulfillment center information, and historical orders are combined into a single dataset.
Unnecessary noise is removed, and the data is prepared for modeling.

2️⃣ Learning Demand Patterns

Instead of using raw data directly, the system creates meaningful features such as:

Previous week demand (lag features)
Moving averages to capture trends
Price-based features like discount
Time-based patterns

These features help the model understand how demand evolves over time.

3️⃣ Model Training

Different machine learning models are trained and compared:

Random Forest
CatBoost
LightGBM

Each model learns how different factors affect food demand.

4️⃣ Model Evaluation

The models are tested using standard regression metrics like:

MAE (average error)
RMSE (error sensitivity to large values)
R² score (how well the model explains demand)

Among all models, LightGBM performed the best, capturing demand patterns more effectively.

5️⃣ Forecasting Future Demand

Once trained, the model predicts future demand for upcoming weeks.

Users can input:

Fulfillment center
Meal ID
Pricing and promotion details
Forecast duration (1, 2, or 4 weeks)

The system then generates future demand predictions step-by-step.

# Key Outcome

The final model achieves:

Strong prediction accuracy,
Reliable short-term forecasting,
Better handling of demand fluctuations.

This shows that machine learning can significantly improve decision-making in food supply chains.

# Tech Stack
Python (core logic)

Pandas & NumPy (data handling)

Scikit-learn, CatBoost, LightGBM (modeling)

Jupyter Notebook (development)

HTML/CSS (frontend interface)

# What Makes This Project Useful
Helps reduce food wastage

Improves inventory planning

Supports data-driven decisions

Provides a simple web interface for predictions

# Future Improvements
Add real-time data integration

Include external factors like weather or holidays

Try deep learning models (LSTM)

Deploy as a complete web application

# Final Note
This project demonstrates how combining time-series features with machine learning models can effectively solve real-world problems like food demand forecasting.
