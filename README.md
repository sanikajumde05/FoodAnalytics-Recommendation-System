# FoodAnalytics-Recommendation-System
An end-to-end data science mini-project that analyzes food-delivery transactions, segments customers, predicts order value, and recommends restaurants through an interactive Flask dashboard.
Data integrity note: The application requires real CSV datasets by default. Synthetic demo data is disabled unless FOOD_DEMO=1 is explicitly set. Never report demo results as findings from the real dataset.

Features
1. Food delivery analytics
- KPI summary: order count, revenue, average order value, unique customers, and unique restaurants.
- Monthly order and revenue trends.
- Top restaurants, cuisines, and states.
- Orders by day of week and selected user demographics.
- Data-quality summary, including duplicate and invalid-record handling.
2. Customer segmentation
- Builds RFM-style customer features from ordering behavior.
- Uses K-Means clustering to group customers.
- Produces segment profiles for interpretation and potential customer-retention strategies.
3. Order-value prediction
- Compares regression models, including a baseline, Random Forest, and gradient-boosting models.
- Reports regression metrics such as R², MAE, and RMSE.
- Provides feature-importance information and an actual-versus-predicted view.
4. Restaurant recommendation
- Uses customer–restaurant interaction history and restaurant attributes.
- Supports cuisine, state, and city filters.
- Evaluates recommendations on held-out interactions.
- Exposes recommendation and evaluation information through the dashboard/API.
Technology stack
- Language: Python
- Backend: Flask
- Data processing: pandas, NumPy
- Machine learning: scikit-learn
- Sparse matrices / similarity: SciPy
- Frontend: HTML, CSS, JavaScript
- Charts: Chart.js
- Notebook: Jupyter
Project structure
fd/
├── app.py
├── pipeline.py
├── requirements.txt
├── Food_Delivery_Analytics.ipynb
├── README.md
├── data/
│   ├── orders.csv
│   ├── restaurant.csv
│   └── users.csv
└── static/
    └── index.html
The data/ folder is expected to contain the three CSV files. File names should include order, restaurant, and user respectively so the loader can identify them.
Dataset setup
The pipeline is designed for the Zomato database dataset referenced by the project:
- Kaggle — Zomato Database
Download the relevant orders, restaurant, and users CSVs and place them in data/.
The loader normalizes column names and maps common alternatives such as r_id to restaurant_id, id to restaurant_id for restaurant data, and sales_qty to sales_quantity. The orders file must provide customer ID, restaurant ID, and sales amount; the restaurant and users files must provide their respective IDs. If startup reports missing columns, inspect the CSV headers and update the column mapping in pipeline.py.
Privacy: Do not commit personal or sensitive user information to a public repository. If the dataset includes names, emails, or passwords, remove those fields when they are not required by the analysis and follow the dataset's license and terms.
Installation and setup (macOS / Linux)
Open a terminal in the project root (the folder containing app.py).
1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
2. Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
3. Add the datasets
Make sure data/ contains the orders, restaurant, and users CSV files.
ls data
4. Start the application
python3 app.py
The first startup loads and cleans the data, segments customers, trains models, and evaluates the recommender. Depending on the dataset and computer, this may take a few minutes. When Flask starts, open:
http://127.0.0.1:5000
If port 5000 is already occupied, change the final app.run(...) port in app.py to an available port, for example 5001, restart the app, and visit http://127.0.0.1:5001.
Keep the terminal open while using the dashboard. Press Ctrl+C to stop the server.
Windows
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
Demo mode (synthetic data only)
If you need to test the interface before obtaining the real CSVs, run:
FOOD_DEMO=1 python3 app.py
This uses generated synthetic data for software testing only. Do not use demo-mode KPIs, model metrics, or recommendations as evidence of real-world performance. To return to real-data mode, stop the app and run python3 app.py without FOOD_DEMO=1.
API endpoints
When the Flask server is running, the application exposes:
Endpoint	Purpose
/	Serves the dashboard
/api/health	Reports application status, data source, demo flag, and order count
/api/analytics	Returns KPI, chart, segment, model, and evaluation data
/api/users	Returns eligible customer IDs for the recommender
/api/filters	Returns cuisine, state, and city filter options
/api/cuisines	Returns available cuisines
/api/recommend/<user_id>	Returns recommendations for a customer


Example recommendation request:
http://127.0.0.1:5000/api/recommend/10001?n=10&cuisine=Chinese
The customer ID and cuisine must exist in the loaded dataset. Query parameters for state and city can also be supplied.
Notebook
To explore the analysis in Jupyter:
jupyter notebook Food_Delivery_Analytics.ipynb
Run the notebook in the same environment where the project dependencies and data are available.
Interpreting results responsibly
- Regression metrics describe the model's performance on its evaluation split; they do not automatically demonstrate future-demand forecasting.
- Recommendation metrics depend on the test split and the definition of a relevant recommendation. Compare models using the same evaluation protocol.
- Customer clusters are exploratory groups, not objectively correct labels. Interpret them using their feature profiles.
- Check the data-quality report, extreme order values, missing ratings, and feature scales before relying on model outputs.
- If numerical warnings such as overflow, divide-by-zero, or invalid matrix operations appear during training, investigate them before presenting model scores as validated.
Troubleshooting
Error: real dataset not found
Add the orders, restaurant, and users CSVs to data/. Confirm the filenames contain order, restaurant, and user.
Error: required columns not found
Check the CSV headers printed in the error. Update ALIASES in pipeline.py if your dataset uses different names.
Error: address already in use
Port 5000 is occupied. Stop the existing Flask process or change the port in app.py.
Dashboard cannot load data
Ensure python3 app.py is still running and open the same port shown in the terminal.
Model warnings about overflow or invalid values
Inspect numeric columns for extremely large values, infinities, missing values, and unscaled features. Fix the underlying data/model issue rather than hiding the warnings.
Limitations and future work
- Validate the models on a separate, appropriate test set and report metrics with clear definitions.
- Consider time-based evaluation where the goal is to predict future behavior.
- Improve handling of sparse customer–restaurant interactions and new users/restaurants.
- Tune and validate recommendation ranking with consistent Precision@K, Recall@K, and NDCG@K metrics.
- Add automated tests, model persistence, configuration for the port, and deployment documentation.
Disclaimer
This project is intended for educational and academic use. Results depend on the dataset, preprocessing choices, feature engineering, and evaluation methodology. No model performance is guaranteed.
