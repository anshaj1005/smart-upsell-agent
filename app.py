import os
import json
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sqlite3
import logging

app = Flask(__name__)
app.secret_key = 'smart-upsell-agent-enhanced-2025'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSmartUpsellAgent:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.customer_segments = None
        self.churn_model = None
        self.feature_columns = [
            'days_since_signup', 'feature_usage_score', 'login_frequency',
            'support_tickets', 'current_plan_tier', 'monthly_usage',
            'user_engagement_score', 'previous_upgrades'
        ]
        self.load_models()

    def load_models(self):
        try:
            with open('models/upsell_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Models loaded successfully")
        except FileNotFoundError:
            logger.info("Training new models...")
            self.train_all_models()

    def generate_enhanced_synthetic_data(self, n_samples=2000):
        np.random.seed(42)
        data = []

        for i in range(n_samples):
            days_since_signup = np.random.randint(1, 730)
            feature_usage_score = np.random.uniform(0, 100)
            login_frequency = np.random.randint(1, 45)
            support_tickets = np.random.randint(0, 15)
            current_plan_tier = np.random.randint(1, 4)
            monthly_usage = np.random.uniform(10, 2000)
            user_engagement_score = np.random.uniform(0, 100)
            previous_upgrades = np.random.randint(0, 5)

            session_duration_avg = np.random.uniform(5, 180)
            pages_per_session = np.random.uniform(1, 50)
            feature_adoption_rate = np.random.uniform(0, 1)

            upsell_probability = (
                (feature_usage_score / 100) * 0.25 +
                (min(login_frequency, 30) / 30) * 0.20 +
                (min(monthly_usage, 1000) / 1000) * 0.15 +
                (user_engagement_score / 100) * 0.15 +
                (previous_upgrades / 3) * 0.10 +
                (min(session_duration_avg, 60) / 60) * 0.10 +
                (feature_adoption_rate) * 0.05
            )

            churn_probability = 1 - (
                (feature_usage_score / 100) * 0.30 +
                (min(login_frequency, 20) / 20) * 0.25 +
                (user_engagement_score / 100) * 0.25 +
                (1 - support_tickets / 10) * 0.20
            )

            upsell_probability += np.random.normal(0, 0.1)
            churn_probability += np.random.normal(0, 0.1)

            upsell_target = 1 if upsell_probability > 0.6 else 0
            churn_target = 1 if churn_probability > 0.5 else 0

            data.append({
                'user_id': f'user_{i+1}',
                'days_since_signup': days_since_signup,
                'feature_usage_score': feature_usage_score,
                'login_frequency': login_frequency,
                'support_tickets': support_tickets,
                'current_plan_tier': current_plan_tier,
                'monthly_usage': monthly_usage,
                'user_engagement_score': user_engagement_score,
                'previous_upgrades': previous_upgrades,
                'session_duration_avg': session_duration_avg,
                'pages_per_session': pages_per_session,
                'feature_adoption_rate': feature_adoption_rate,
                'upsell_target': upsell_target,
                'churn_target': churn_target
            })

        return pd.DataFrame(data)

    def train_all_models(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        df = self.generate_enhanced_synthetic_data()

        X_upsell = df[self.feature_columns]
        y_upsell = df['upsell_target']

        X_train, X_test, y_train, y_test = train_test_split(X_upsell, y_upsell, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        extended_features = self.feature_columns + ['session_duration_avg', 'pages_per_session', 'feature_adoption_rate']
        X_churn = df[extended_features]
        y_churn = df['churn_target']

        self.churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.churn_model.fit(X_churn, y_churn)

        segmentation_features = ['feature_usage_score', 'login_frequency', 'user_engagement_score', 'monthly_usage']
        X_segments = df[segmentation_features]

        silhouette_scores = []
        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_segments)
            silhouette_avg = silhouette_score(X_segments, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        self.customer_segments = KMeans(n_clusters=optimal_k, random_state=42)
        self.customer_segments.fit(X_segments)

        os.makedirs('models', exist_ok=True)
        with open('models/upsell_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('models/churn_model.pkl', 'wb') as f:
            pickle.dump(self.churn_model, f)
        with open('models/segments.pkl', 'wb') as f:
            pickle.dump(self.customer_segments, f)

        upsell_pred = self.model.predict(X_test_scaled)
        upsell_accuracy = accuracy_score(y_test, upsell_pred)

        logger.info(f"Enhanced Model Accuracy: {upsell_accuracy:.2f}")
        logger.info("All enhanced models trained successfully")

    def predict_upsell_probability(self, user_data):
        if self.model is None or self.scaler is None:
            return 0.0

        features = [user_data.get(col, 0) for col in self.feature_columns]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        probability = self.model.predict_proba(features_scaled)[0][1]
        return probability

    def predict_churn_risk(self, user_data):
        if self.churn_model is None:
            return 0.0

        extended_features = self.feature_columns + ['session_duration_avg', 'pages_per_session', 'feature_adoption_rate']
        features = []
        for col in extended_features:
            if col in ['session_duration_avg', 'pages_per_session', 'feature_adoption_rate']:
                if col == 'session_duration_avg':
                    features.append(user_data.get(col, 30))
                elif col == 'pages_per_session':
                    features.append(user_data.get(col, 5))
                else:
                    features.append(user_data.get(col, 0.5))
            else:
                features.append(user_data.get(col, 0))

        features_array = np.array(features).reshape(1, -1)
        probability = self.churn_model.predict_proba(features_array)[0][1]
        return probability

    def get_customer_segment(self, user_data):
        if self.customer_segments is None:
            return 0

        segmentation_features = ['feature_usage_score', 'login_frequency', 'user_engagement_score', 'monthly_usage']
        features = [user_data.get(col, 0) for col in segmentation_features]
        features_array = np.array(features).reshape(1, -1)
        segment = self.customer_segments.predict(features_array)[0]
        return segment

    def generate_advanced_message(self, user_data, probability, churn_risk=None, segment=None):
        user_name = user_data.get('name', 'Valued Customer')

        segment_names = {0: "Power Users", 1: "Growing Teams", 2: "Casual Users", 3: "Enterprise Ready", 4: "New Adopters"}
        segment_name = segment_names.get(segment, "Valued Customer")

        if churn_risk and churn_risk > 0.7:
            if probability > 0.8:
                urgency = "critical"
                message = f"Hi {user_name}! We value our {segment_name} - upgrade now and get 40% off for 3 months!"
            else:
                urgency = "high"
                message = f"Hi {user_name}! Let us help you succeed with our {self.get_next_plan(user_data.get('current_plan_tier', 1))} features."
        elif probability > 0.8:
            urgency = "high"
            message = f"Hi {user_name}! Perfect time to upgrade to {self.get_next_plan(user_data.get('current_plan_tier', 1))} - 30% off for {segment_name}!"
        elif probability > 0.6:
            urgency = "medium"
            message = f"Hello {user_name}! {segment_name} love our {self.get_next_plan(user_data.get('current_plan_tier', 1))} features. Ready to scale?"
        else:
            urgency = "low"
            message = f"Hi {user_name}! Discover how {self.get_next_plan(user_data.get('current_plan_tier', 1))} helps {segment_name} achieve more."

        return {
            'message': message,
            'urgency': urgency,
            'recommended_plan': self.get_next_plan(user_data.get('current_plan_tier', 1)),
            'probability': probability,
            'segment': segment_name,
            'churn_risk': churn_risk if churn_risk else 0.0
        }

    def get_plan_name(self, tier):
        plans = {1: 'Basic Plan', 2: 'Professional Plan', 3: 'Enterprise Plan', 4: 'Enterprise Plus'}
        return plans.get(tier, 'Basic Plan')

    def get_next_plan(self, current_tier):
        next_plans = {1: 'Professional Plan', 2: 'Enterprise Plan', 3: 'Enterprise Plus', 4: 'Custom Plan'}
        return next_plans.get(current_tier, 'Professional Plan')

upsell_agent = EnhancedSmartUpsellAgent()

def init_enhanced_db():
    conn = sqlite3.connect('data/users.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            name TEXT,
            email TEXT,
            signup_date TEXT,
            current_plan_tier INTEGER,
            feature_usage_score REAL,
            login_frequency INTEGER,
            support_tickets INTEGER,
            monthly_usage REAL,
            user_engagement_score REAL,
            previous_upgrades INTEGER,
            last_upsell_date TEXT,
            session_duration_avg REAL DEFAULT 30,
            pages_per_session REAL DEFAULT 5,
            feature_adoption_rate REAL DEFAULT 0.5,
            customer_segment INTEGER DEFAULT 0,
            churn_risk REAL DEFAULT 0.0,
            ltv_prediction REAL DEFAULT 0.0,
            last_activity_date TEXT,
            conversion_funnel_stage INTEGER DEFAULT 1
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS upsell_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            probability REAL,
            message TEXT,
            recommended_plan TEXT,
            status TEXT,
            channel TEXT DEFAULT 'in-app',
            segment TEXT,
            churn_risk REAL,
            conversion_result TEXT DEFAULT 'pending',
            revenue_impact REAL DEFAULT 0.0,
            campaign_id TEXT
        )
    ''')

    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics_dashboard():
    return render_template('analytics.html')

@app.route('/campaigns') 
def campaign_manager():
    return render_template('campaigns.html')

@app.route('/segments')
def segments_view():
    return render_template('segments.html')

@app.route('/api/users')
def get_users():
    conn = sqlite3.connect('data/users.db')
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()

    users_data = []
    for _, user in df.iterrows():
        user_dict = user.to_dict()
        signup_date = datetime.fromisoformat(user_dict['signup_date'])
        user_dict['days_since_signup'] = (datetime.now() - signup_date).days

        upsell_probability = upsell_agent.predict_upsell_probability(user_dict)
        churn_risk = upsell_agent.predict_churn_risk(user_dict)
        segment = upsell_agent.get_customer_segment(user_dict)

        user_dict['upsell_probability'] = round(upsell_probability * 100, 2)
        user_dict['churn_risk'] = round(churn_risk * 100, 2)
        user_dict['customer_segment'] = segment

        recommendation = upsell_agent.generate_advanced_message(
            user_dict, upsell_probability, churn_risk, segment
        )
        user_dict['recommendation'] = recommendation

        plan_values = {1: 50, 2: 150, 3: 500, 4: 1000}
        base_ltv = plan_values.get(user_dict.get('current_plan_tier', 1), 50)
        ltv_multiplier = 1 + (upsell_probability * 2) + (user_dict['user_engagement_score'] / 100)
        user_dict['ltv_prediction'] = round(base_ltv * ltv_multiplier, 2)

        users_data.append(user_dict)

    return jsonify(users_data)

@app.route('/api/user/<user_id>/upsell', methods=['POST'])
def send_enhanced_upsell(user_id):
    data = request.get_json() or {}
    channel = data.get('channel', 'in-app')
    campaign_id = data.get('campaign_id', 'manual')

    conn = sqlite3.connect('data/users.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    columns = [description[0] for description in cursor.description]
    user_dict = dict(zip(columns, user))

    signup_date = datetime.fromisoformat(user_dict['signup_date'])
    user_dict['days_since_signup'] = (datetime.now() - signup_date).days

    probability = upsell_agent.predict_upsell_probability(user_dict)
    churn_risk = upsell_agent.predict_churn_risk(user_dict)
    segment = upsell_agent.get_customer_segment(user_dict)

    recommendation = upsell_agent.generate_advanced_message(
        user_dict, probability, churn_risk, segment
    )

    plan_values = {1: 50, 2: 150, 3: 500}
    current_value = plan_values.get(user_dict.get('current_plan_tier', 1), 50)
    next_tier_value = plan_values.get(user_dict.get('current_plan_tier', 1) + 1, 150)
    revenue_impact = (next_tier_value - current_value) * probability

    cursor.execute('''
        INSERT INTO upsell_history 
        (user_id, timestamp, probability, message, recommended_plan, status, 
         channel, segment, churn_risk, conversion_result, revenue_impact, campaign_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, datetime.now().isoformat(), probability,
        recommendation['message'], recommendation['recommended_plan'], 'sent',
        channel, recommendation.get('segment', ''), churn_risk, 'pending',
        revenue_impact, campaign_id
    ))

    cursor.execute(
        "UPDATE users SET last_upsell_date = ?, churn_risk = ?, customer_segment = ?, last_activity_date = ? WHERE user_id = ?",
        (datetime.now().isoformat(), churn_risk, segment, datetime.now().isoformat(), user_id)
    )

    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'message': f'Enhanced AI upsell sent via {channel}',
        'recommendation': recommendation,
        'revenue_impact': round(revenue_impact, 2),
        'insights': {
            'segment': recommendation.get('segment'),
            'churn_risk_level': 'High' if churn_risk > 0.7 else 'Medium' if churn_risk > 0.4 else 'Low',
            'success_probability': f"{round(probability * 100)}%"
        }
    })

@app.route('/api/enhanced-analytics')
def get_enhanced_analytics():
    conn = sqlite3.connect('data/users.db')
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    history_df = pd.read_sql_query("SELECT * FROM upsell_history", conn)

    total_users = len(users_df)
    high_value_users = len(users_df[users_df['current_plan_tier'] >= 2]) if not users_df.empty else 0

    if not users_df.empty:
        avg_churn_risk = users_df['churn_risk'].mean() if 'churn_risk' in users_df.columns else 0
        high_churn_users = len(users_df[users_df['churn_risk'] > 70]) if 'churn_risk' in users_df.columns else 0
        avg_ltv = users_df['ltv_prediction'].mean() if 'ltv_prediction' in users_df.columns else 0
    else:
        avg_churn_risk = 0
        high_churn_users = 0
        avg_ltv = 0

    segment_distribution = {}
    if 'customer_segment' in users_df.columns and not users_df.empty:
        segment_counts = users_df['customer_segment'].value_counts().to_dict()
        segment_names = {0: "Power Users", 1: "Growing Teams", 2: "Casual Users", 3: "Enterprise Ready", 4: "New Adopters"}
        segment_distribution = {segment_names.get(k, f"Segment {k}"): v for k, v in segment_counts.items()}

    revenue_impact = history_df['revenue_impact'].sum() if 'revenue_impact' in history_df.columns and not history_df.empty else 0

    total_campaigns = len(history_df)
    simulated_conversions = int(total_campaigns * 0.235)

    analytics = {
        'total_users': total_users,
        'total_upsells_sent': total_campaigns,
        'high_value_users': high_value_users,
        'avg_churn_risk': round(avg_churn_risk, 2),
        'high_churn_users': high_churn_users,
        'avg_ltv': round(avg_ltv, 2),
        'segment_distribution': segment_distribution,
        'plan_distribution': users_df['current_plan_tier'].value_counts().to_dict() if not users_df.empty else {},
        'recent_activity': history_df.tail(20).to_dict('records') if not history_df.empty else [],
        'conversion_rate': 23.5,
        'revenue_impact': round(revenue_impact, 2),
        'total_conversions': simulated_conversions,
        'roi': 3.2
    }

    conn.close()
    return jsonify(analytics)

@app.route('/api/segments-analysis')
def get_segments_analysis():
    conn = sqlite3.connect('data/users.db')
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()

    if users_df.empty:
        return jsonify({'error': 'No user data available'})

    segment_names = {0: "Power Users", 1: "Growing Teams", 2: "Casual Users", 3: "Enterprise Ready", 4: "New Adopters"}
    segment_analysis = {}

    for segment_id, segment_name in segment_names.items():
        segment_users = users_df[users_df['customer_segment'] == segment_id]

        if not segment_users.empty:
            segment_analysis[segment_name] = {
                'count': len(segment_users),
                'avg_engagement': round(segment_users['user_engagement_score'].mean(), 2),
                'avg_usage': round(segment_users['monthly_usage'].mean(), 2),
                'avg_churn_risk': round(segment_users['churn_risk'].mean(), 2),
                'plan_distribution': segment_users['current_plan_tier'].value_counts().to_dict(),
                'characteristics': {
                    'high_engagement': len(segment_users[segment_users['user_engagement_score'] > 80]),
                    'frequent_users': len(segment_users[segment_users['login_frequency'] > 20]),
                    'power_features': len(segment_users[segment_users['feature_usage_score'] > 75])
                }
            }

    return jsonify(segment_analysis)

@app.route('/add_enhanced_sample_data')
def add_enhanced_sample_data():
    sample_users = [
        {
            'user_id': 'user_demo_1', 'name': 'Alice Johnson', 'email': 'alice@techstartup.com',
            'signup_date': (datetime.now() - timedelta(days=45)).isoformat(), 'current_plan_tier': 1,
            'feature_usage_score': 85.2, 'login_frequency': 18, 'support_tickets': 2,
            'monthly_usage': 450.0, 'user_engagement_score': 78.5, 'previous_upgrades': 0,
            'session_duration_avg': 45.2, 'pages_per_session': 8.3, 'feature_adoption_rate': 0.75
        },
        {
            'user_id': 'user_demo_2', 'name': 'Bob Smith', 'email': 'bob@scalecorp.com',
            'signup_date': (datetime.now() - timedelta(days=120)).isoformat(), 'current_plan_tier': 2,
            'feature_usage_score': 65.8, 'login_frequency': 12, 'support_tickets': 1,
            'monthly_usage': 280.0, 'user_engagement_score': 82.3, 'previous_upgrades': 1,
            'session_duration_avg': 32.1, 'pages_per_session': 6.2, 'feature_adoption_rate': 0.60
        },
        {
            'user_id': 'user_demo_3', 'name': 'Carol Davis', 'email': 'carol@innovatetech.com',
            'signup_date': (datetime.now() - timedelta(days=30)).isoformat(), 'current_plan_tier': 1,
            'feature_usage_score': 92.1, 'login_frequency': 25, 'support_tickets': 0,
            'monthly_usage': 680.0, 'user_engagement_score': 95.2, 'previous_upgrades': 0,
            'session_duration_avg': 62.4, 'pages_per_session': 12.1, 'feature_adoption_rate': 0.90
        },
        {
            'user_id': 'user_demo_4', 'name': 'David Wilson', 'email': 'david@businesspro.com',
            'signup_date': (datetime.now() - timedelta(days=200)).isoformat(), 'current_plan_tier': 3,
            'feature_usage_score': 45.3, 'login_frequency': 8, 'support_tickets': 3,
            'monthly_usage': 150.0, 'user_engagement_score': 55.1, 'previous_upgrades': 2,
            'session_duration_avg': 18.7, 'pages_per_session': 3.8, 'feature_adoption_rate': 0.35
        },
        {
            'user_id': 'user_demo_5', 'name': 'Emma Rodriguez', 'email': 'emma@growthco.com',
            'signup_date': (datetime.now() - timedelta(days=15)).isoformat(), 'current_plan_tier': 1,
            'feature_usage_score': 73.4, 'login_frequency': 22, 'support_tickets': 1,
            'monthly_usage': 385.0, 'user_engagement_score': 88.7, 'previous_upgrades': 0,
            'session_duration_avg': 38.9, 'pages_per_session': 7.5, 'feature_adoption_rate': 0.68
        },
        {
            'user_id': 'user_demo_6', 'name': 'Frank Chen', 'email': 'frank@aiventures.com',
            'signup_date': (datetime.now() - timedelta(days=90)).isoformat(), 'current_plan_tier': 2,
            'feature_usage_score': 88.9, 'login_frequency': 28, 'support_tickets': 0,
            'monthly_usage': 720.0, 'user_engagement_score': 91.3, 'previous_upgrades': 1,
            'session_duration_avg': 55.3, 'pages_per_session': 11.2, 'feature_adoption_rate': 0.85
        }
    ]

    conn = sqlite3.connect('data/users.db')
    cursor = conn.cursor()

    for user in sample_users:
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, name, email, signup_date, current_plan_tier, feature_usage_score,
             login_frequency, support_tickets, monthly_usage, user_engagement_score,
             previous_upgrades, last_upsell_date, session_duration_avg, pages_per_session, 
             feature_adoption_rate, last_activity_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user['user_id'], user['name'], user['email'], user['signup_date'],
            user['current_plan_tier'], user['feature_usage_score'], user['login_frequency'],
            user['support_tickets'], user['monthly_usage'], user['user_engagement_score'],
            user['previous_upgrades'], None, user['session_duration_avg'], 
            user['pages_per_session'], user['feature_adoption_rate'], datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'message': 'Enhanced sample data loaded successfully!'})

if __name__ == '__main__':
    init_enhanced_db()
    print("🚀 ENHANCED Smart Upsell Agent Starting...")
    print("📊 Dashboard: http://localhost:5000")
    print("📈 Analytics: http://localhost:5000/analytics")
    print("📧 Campaigns: http://localhost:5000/campaigns")
    print("🎯 Segments: http://localhost:5000/segments")
    app.run(debug=True, host='0.0.0.0', port=5000)
