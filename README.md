# Food Delivery Analytics & Recommendation System

A machine learning project that analyzes food-delivery data to identify sales trends, segment customers, forecast demand, and recommend restaurants through an interactive dashboard.

## Features
- Sales and restaurant analytics
- Customer segmentation using RFM and K-Means
- Demand forecasting using Gradient Boosting
- Recommendation models: Popularity, Content-Based, Item-KNN, SVD, and Hybrid
- Interactive dashboard built with Flask

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Flask, HTML, CSS, JavaScript, Matplotlib, Seaborn

## Run Locally
```bash
git clone <your-repository-url>
cd <project-folder>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Dataset
Place `orders.csv`, `restaurant.csv`, and `users.csv` inside the `data/` folder.

## Key Insight
Popularity achieved the best reported ranking performance, while the hybrid recommender provided broader catalogue coverage. Results depend on data quality and should be interpreted in context.
