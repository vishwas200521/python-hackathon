import numpy as np
import pandas as pd
import datetime
import random
import math
# Add this at the top of the file
import sys
import codecs
from collections import defaultdict, deque

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())




class NeuralNetwork:
    """Neural Network for advanced pattern recognition"""
    def __init__(self, input_size=8, hidden_size=12, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = 0.01
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Backward propagation
        dz2 = output - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def fit(self, X, y, epochs=100):
        X = np.array(X)
        y = np.array(y)
        
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 25 == 0:
                loss = np.mean((output.flatten() - y) ** 2)
                print(f"  Neural Network - Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        X = np.array(X)
        output = self.forward(X)
        return np.maximum(0, output.flatten()).tolist()


class RandomForestRegressor:
    """Random Forest implementation for robust predictions"""
    def __init__(self, n_trees=10, max_depth=6, min_samples=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
    
    def bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return [X[i] for i in indices], [y[i] for i in indices]
    
    def fit(self, X, y):
        print(f"  Training Random Forest with {self.n_trees} trees...")
        self.trees = []
        
        for i in range(self.n_trees):
            X_boot, y_boot = self.bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
    
    def predict(self, X):
        if not self.trees:
            return [0] * len(X)
        
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.maximum(0, np.mean(predictions, axis=0)).tolist()


class DecisionTreeRegressor:
    """Decision Tree for Random Forest"""
    def __init__(self, max_depth=5, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
    
    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]
    
    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            return np.mean(y)
        
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)
        
        left_indices = [i for i, x in enumerate(X) if x[best_feature] <= best_threshold]
        right_indices = [i for i, x in enumerate(X) if x[best_feature] > best_threshold]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return np.mean(y)
        
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }
    
    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(len(X[0])):
            values = sorted(set(x[feature] for x in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                gain = self._calculate_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_gain(self, X, y, feature, threshold):
        left_y = [y[i] for i, x in enumerate(X) if x[feature] <= threshold]
        right_y = [y[i] for i, x in enumerate(X) if x[feature] > threshold]
        
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        total_var = np.var(y)
        left_var = np.var(left_y) if len(left_y) > 0 else 0
        right_var = np.var(right_y) if len(right_y) > 0 else 0
        
        weighted_var = (len(left_y) * left_var + len(right_y) * right_var) / len(y)
        return total_var - weighted_var
    
    def _predict_sample(self, x, node):
        if isinstance(node, (int, float)):
            return node
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])


class GradientBoostingRegressor:
    """Gradient Boosting for advanced ensemble learning"""
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_prediction = 0
    
    def fit(self, X, y):
        print(f"  Training Gradient Boosting with {self.n_estimators} estimators...")
        y = np.array(y)
        self.base_prediction = np.mean(y)
        
        predictions = np.full(len(y), self.base_prediction)
        
        for i in range(self.n_estimators):
            residuals = y - predictions
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples=3)
            tree.fit(X, residuals.tolist())
            
            tree_predictions = np.array(tree.predict(X))
            predictions += self.learning_rate * tree_predictions
            
            self.trees.append(tree)
            
            if i % 10 == 0:
                mse = np.mean((y - predictions) ** 2)
                print(f"    Iteration {i}, MSE: {mse:.4f}")
    
    def predict(self, X):
        predictions = np.full(len(X), self.base_prediction)
        
        for tree in self.trees:
            tree_pred = np.array(tree.predict(X))
            predictions += self.learning_rate * tree_pred
        
        return np.maximum(0, predictions).tolist()


class DailyPassengerTracker:
    """Track day-to-day passenger flows and generate insights"""
    def __init__(self):
        self.daily_data = defaultdict(dict)
        self.station_trends = defaultdict(list)
        self.route_analytics = defaultdict(dict)
        self.growth_patterns = {}
    
    def add_daily_record(self, date, station, entered, exited, route, weather_data=None):
        """Add daily passenger record"""
        if date not in self.daily_data:
            self.daily_data[date] = {}
        
        self.daily_data[date][station] = {
            'entered': entered,
            'exited': exited,
            'total': entered + exited,
            'net_flow': entered - exited,
            'route': route,
            'weather': weather_data or {}
        }
        
        # Update station trends
        self.station_trends[station].append({
            'date': date,
            'total': entered + exited,
            'entered': entered,
            'exited': exited
        })
        
        # Keep only last 30 days for trends
        if len(self.station_trends[station]) > 30:
            self.station_trends[station] = self.station_trends[station][-30:]
    
    def get_daily_summary(self, date):
        """Get summary for specific date"""
        if date not in self.daily_data:
            return None
        
        day_data = self.daily_data[date]
        total_entered = sum(station['entered'] for station in day_data.values())
        total_exited = sum(station['exited'] for station in day_data.values())
        
        busiest_station = max(day_data.keys(), key=lambda s: day_data[s]['total'])
        
        return {
            'date': date,
            'total_entered': total_entered,
            'total_exited': total_exited,
            'total_passengers': total_entered + total_exited,
            'net_flow': total_entered - total_exited,
            'busiest_station': busiest_station,
            'station_count': len(day_data),
            'stations': list(day_data.keys())
        }
    
    def get_station_trend(self, station, days=7):
        """Get trend analysis for station"""
        if station not in self.station_trends:
            return None
        
        recent_data = self.station_trends[station][-days:]
        if len(recent_data) < 2:
            return None
        
        recent_avg = np.mean([d['total'] for d in recent_data[-3:]])
        earlier_avg = np.mean([d['total'] for d in recent_data[:3]])
        
        trend_pct = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0
        
        return {
            'station': station,
            'days_analyzed': len(recent_data),
            'recent_average': recent_avg,
            'earlier_average': earlier_avg,
            'trend_percentage': trend_pct,
            'trend_direction': 'increasing' if trend_pct > 5 else 'decreasing' if trend_pct < -5 else 'stable',
            'daily_totals': [d['total'] for d in recent_data]
        }
    
    def predict_next_day(self, station, weather_factor=1.0):
        """Predict next day passenger count"""
        if station not in self.station_trends:
            return None
        
        recent_data = self.station_trends[station][-7:]  # Last 7 days
        if len(recent_data) < 3:
            return None
        
        # Simple trend-based prediction
        daily_totals = [d['total'] for d in recent_data]
        
        # Calculate trend
        if len(daily_totals) >= 3:
            trend = (daily_totals[-1] - daily_totals[0]) / len(daily_totals)
        else:
            trend = 0
        
        # Base prediction on recent average
        base_prediction = np.mean(daily_totals[-3:])
        
        # Apply trend and weather factor
        prediction = base_prediction + trend
        prediction *= weather_factor
        
        return max(0, int(prediction))


class KMRLAdvancedAI:
    """Advanced KMRL AI System with Multiple Models"""
    def __init__(self):
        self.models = {}
        self.passenger_tracker = DailyPassengerTracker()
        self.training_data = None
        
        # KMRL Route Configuration
        self.routes = {
            'aluva_pettah': {
                'stations': ['Aluva', 'Pulinchodu', 'Companypadi', 'Ambattukavu', 'Muttom', 
                           'Kalamassery', 'CUSAT', 'Pathadipalam', 'Edapally', 'Changampuzha Park',
                           'Palarivattom', 'JLN Stadium', 'Kaloor', 'Town Hall', 'MG Road',
                           'Maharajas', 'Ernakulam South', 'Kadavanthra', 'Elamkulam', 'Vyttila',
                           'Thaikoodam', 'Pettah'],
                'distance_km': 25.6,
                'travel_time_min': 45
            },
            'jln_pettah': {
                'stations': ['JLN Stadium', 'Kaloor', 'Town Hall', 'MG Road', 'Maharajas',
                           'Ernakulam South', 'Kadavanthra', 'Elamkulam', 'Vyttila', 
                           'Thaikoodam', 'Pettah'],
                'distance_km': 11.2,
                'travel_time_min': 22
            }
        }
        
        # Initialize multiple AI models
        self.setup_models()
    
    def setup_models(self):
        """Initialize all AI models"""
        print("[SETUP] Initializing multiple AI models...")
        
        self.models = {
            'neural_network': NeuralNetwork(input_size=8, hidden_size=15),
            'random_forest': RandomForestRegressor(n_trees=12, max_depth=7),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=30, learning_rate=0.1),
            'ensemble_weights': {'neural_network': 0.4, 'random_forest': 0.35, 'gradient_boosting': 0.25}
        }
        
        print("  ✓ Neural Network initialized")
        print("  ✓ Random Forest initialized")
        print("  ✓ Gradient Boosting initialized")
    
    def load_data_from_csv(self, file_path):
        """Load training data from CSV with enhanced processing"""
        print(f"[DATA] Loading from {file_path}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        except Exception as e:
            print(f"  Error loading CSV: {e}")
            return self.generate_synthetic_data(days=90)
        
        data = []
        
        for _, row in df.iterrows():
            try:
                # Enhanced data processing
                processed_row = {
                    'date': str(row['date']),
                    'hour': int(row.get('hour', 8)),
                    'day_of_week': self._process_day_of_week(row.get('day_of_week', 0)),
                    'route': self._process_route(row.get('route', 'aluva_pettah')),
                    'ridership': max(0, int(row.get('ridership', 0))),
                    'delay_minutes': max(0, float(row.get('delay_minutes', 0))),
                    'weather_factor': max(0.1, min(2.0, float(row.get('weather_factor', 1.0)))),
                    'is_peak': int(row.get('is_peak', 0)),
                    'is_weekend': int(row.get('is_weekend', 0)),
                    'frequency_minutes': max(2, min(15, int(row.get('frequency_minutes', 6))))
                }
                
                # Add derived features
                processed_row['month'] = self._extract_month(processed_row['date'])
                processed_row['season'] = self._get_season(processed_row['month'])
                
                data.append(processed_row)
                
                # Update daily passenger tracker
                if 'station' in row and pd.notna(row['station']):
                    self.passenger_tracker.add_daily_record(
                        date=processed_row['date'],
                        station=str(row['station']),
                        entered=processed_row['ridership'] // 2,
                        exited=processed_row['ridership'] // 2,
                        route=processed_row['route'],
                        weather_data={'factor': processed_row['weather_factor']}
                    )
                
            except Exception as e:
                print(f"  Warning: Error processing row {len(data)}: {e}")
                continue
        
        print(f"  Successfully processed {len(data)} records")
        return data
    
    def _process_day_of_week(self, value):
        """Process day of week value"""
        if isinstance(value, str):
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            return days.index(value.lower()) if value.lower() in days else 0
        return int(value) if pd.notna(value) else 0
    
    def _process_route(self, value):
        """Process route value"""
        if pd.isna(value):
            return 'aluva_pettah'
        
        route_str = str(value).lower().replace("–", "_").replace("'", "").replace(" ", "_")
        
        if 'aluva' in route_str or 'pettah' in route_str:
            return 'aluva_pettah'
        elif 'jln' in route_str:
            return 'jln_pettah'
        else:
            return route_str
    
    def _extract_month(self, date_str):
        """Extract month from date string"""
        try:
            if isinstance(date_str, str):
                return datetime.datetime.strptime(date_str, '%Y-%m-%d').month
            return 1
        except:
            return 1
    
    def _get_season(self, month):
        """Get season from month (for Indian climate)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Summer
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon
        else:
            return 3  # Post-monsoon
    
    def train_all_models(self, data):
        """Train all AI models with the dataset"""
        print(f"\n[TRAINING] Training multiple AI models with {len(data)} records...")
        
        self.training_data = data
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(data)
        
        print(f"  Prepared {len(X_train)} training samples with {len(X_train[0])} features")
        
        # Train Neural Network
        print("  Training Neural Network...")
        self.models['neural_network'].fit(X_train, y_train, epochs=80)
        
        # Train Random Forest
        print("  Training Random Forest...")
        self.models['random_forest'].fit(X_train, y_train)
        
        # Train Gradient Boosting
        print("  Training Gradient Boosting...")
        self.models['gradient_boosting'].fit(X_train, y_train)
        
        print("✓ All models trained successfully!")
        
        # Evaluate models
        self._evaluate_models(X_train[:100], y_train[:100])
    
    def _prepare_training_data(self, data):
        """Prepare data for training"""
        X_train = []
        y_train = []
        
        for record in data:
            features = [
                record['hour'],
                record['day_of_week'],
                record['weather_factor'],
                int(record['is_peak']),
                int(record['is_weekend']),
                record['month'],
                record['season'],
                1 if record['route'] == 'aluva_pettah' else 0
            ]
            
            X_train.append(features)
            y_train.append(record['ridership'])
        
        return X_train, y_train
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n[EVALUATION] Model Performance:")
        
        for model_name in ['neural_network', 'random_forest', 'gradient_boosting']:
            try:
                predictions = self.models[model_name].predict(X_test)
                mae = np.mean([abs(p - a) for p, a in zip(predictions, y_test)])
                rmse = np.sqrt(np.mean([(p - a)**2 for p, a in zip(predictions, y_test)]))
                
                print(f"  {model_name.title().replace('_', ' ')}:")
                print(f"    MAE: {mae:.2f}")
                print(f"    RMSE: {rmse:.2f}")
                
            except Exception as e:
                print(f"  {model_name}: Evaluation failed - {e}")
    
    def predict_ridership_ensemble(self, date_time, route, weather_factor=1.0):
        """Predict ridership using ensemble of models"""
        # Prepare features
        features = [
            date_time.hour,
            date_time.weekday(),
            weather_factor,
            int(7 <= date_time.hour <= 9 or 17 <= date_time.hour <= 19),  # is_peak
            int(date_time.weekday() >= 5),  # is_weekend
            date_time.month,
            self._get_season(date_time.month),
            1 if route == 'aluva_pettah' else 0
        ]
        
        predictions = {}
        
        # Get predictions from all models
        for model_name in ['neural_network', 'random_forest', 'gradient_boosting']:
            try:
                pred = self.models[model_name].predict([features])[0]
                predictions[model_name] = max(0, pred)
            except Exception as e:
                print(f"  {model_name} prediction failed: {e}")
                predictions[model_name] = 0
        
        # Ensemble prediction
        weights = self.models['ensemble_weights']
        ensemble_pred = sum(predictions[model] * weights[model] for model in predictions.keys())
        
        return {
            'ensemble_prediction': int(ensemble_pred),
            'individual_predictions': predictions,
            'confidence_score': self._calculate_confidence(predictions)
        }
    
    def _calculate_confidence(self, predictions):
        """Calculate prediction confidence based on model agreement"""
        if not predictions:
            return 0.0
        
        values = list(predictions.values())
        if len(values) < 2:
            return 0.5
        
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        # Lower standard deviation = higher confidence
        confidence = max(0.1, min(1.0, 1.0 - (std_dev / (mean_val + 1))))
        return confidence
    
    def get_daily_passenger_insights(self, date):
        """Get comprehensive daily passenger insights"""
        summary = self.passenger_tracker.get_daily_summary(date)
        
        if not summary:
            return f"No data available for {date}"
        
        insights = {
            'date': date,
            'passenger_metrics': {
                'total_passengers': summary['total_passengers'],
                'passengers_entered': summary['total_entered'],
                'passengers_exited': summary['total_exited'],
                'net_passenger_flow': summary['net_flow']
            },
            'operational_metrics': {
                'busiest_station': summary['busiest_station'],
                'active_stations': summary['station_count'],
                'stations_list': summary['stations']
            }
        }
        
        # Add trend analysis for busiest station
        trend = self.passenger_tracker.get_station_trend(summary['busiest_station'])
        if trend:
            insights['trend_analysis'] = {
                'station': trend['station'],
                'trend_direction': trend['trend_direction'],
                'trend_percentage': f"{trend['trend_percentage']:+.1f}%",
                'recent_average': f"{trend['recent_average']:.0f} passengers/day"
            }
        
        return insights
    
    def predict_next_week_passengers(self, station, weather_factors=None):
        """Predict passenger counts for next 7 days"""
        if weather_factors is None:
            weather_factors = [1.0] * 7
        
        predictions = []
        
        for day in range(7):
            weather_factor = weather_factors[day] if day < len(weather_factors) else 1.0
            predicted_count = self.passenger_tracker.predict_next_day(station, weather_factor)
            
            if predicted_count is not None:
                predictions.append({
                    'day': day + 1,
                    'predicted_passengers': predicted_count,
                    'weather_factor': weather_factor,
                    'confidence': 0.8 - (day * 0.1)  # Decreasing confidence over time
                })
            else:
                predictions.append({
                    'day': day + 1,
                    'predicted_passengers': 0,
                    'weather_factor': weather_factor,
                    'confidence': 0.0
                })
        
        return {
            'station': station,
            'predictions': predictions,
            'total_week_forecast': sum(p['predicted_passengers'] for p in predictions)
        }
    
    def generate_synthetic_data(self, days=90):
        """Generate synthetic data for testing"""
        print(f"[SYNTHETIC] Generating {days} days of synthetic data...")
        
        data = []
        start_date = datetime.datetime(2025, 8, 1)
        
        for day in range(days):
            current_date = start_date + datetime.timedelta(days=day)
            date_str = current_date.strftime('%Y-%m-%d')
            
            for hour in range(5, 24):  # Operating hours
                for route in self.routes.keys():
                    ridership = self._generate_ridership(hour, current_date.weekday(), route)
                    
                    data.append({
                        'date': date_str,
                        'hour': hour,
                        'day_of_week': current_date.weekday(),
                        'route': route,
                        'ridership': ridership,
                        'delay_minutes': max(0, np.random.normal(2, 1)),
                        'weather_factor': max(0.5, min(1.5, np.random.normal(1, 0.2))),
                        'is_peak': int(7 <= hour <= 9 or 17 <= hour <= 19),
                        'is_weekend': int(current_date.weekday() >= 5),
                        'frequency_minutes': 6,
                        'month': current_date.month,
                        'season': self._get_season(current_date.month)
                    })
        
        print(f"  Generated {len(data)} synthetic records")
        return data
    
    def _generate_ridership(self, hour, weekday, route):
        """Generate realistic ridership based on patterns"""
        base = 200 if route == 'aluva_pettah' else 120
        
        # Hour pattern
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            multiplier = 2.5  # Peak
        elif 10 <= hour <= 16:
            multiplier = 1.0  # Normal
        else:
            multiplier = 0.4  # Off-peak
        
        # Weekday pattern
        if weekday >= 5:  # Weekend
            multiplier *= 0.7
        
        ridership = base * multiplier
        return max(50, int(ridership + np.random.normal(0, ridership * 0.2)))


# Main execution function
def main():
    print("*** KMRL Advanced AI System with Multiple Models ***")
    print("=" * 60)
    
    # Initialize the advanced AI system
    ai_system = KMRLAdvancedAI()
    
    # Load data (try CSV first, fallback to synthetic)
    try:
        training_data = ai_system.load_data_from_csv("kochi_metro_august_2025.csv")
    except:
        print("[INFO] CSV not found, using synthetic data for demonstration")
        training_data = ai_system.generate_synthetic_data(days=90)
    
    # Train all AI models
    ai_system.train_all_models(training_data)
    
    # Demonstration of capabilities
    print(f"\n[DEMO] AI System Capabilities Demonstration")
    print("=" * 50)
    
    # 1. Multi-model ridership prediction
    test_datetime = datetime.datetime(2025, 9, 20, 8, 30)  # Friday morning peak
    prediction_result = ai_system.predict_ridership_ensemble(
        date_time=test_datetime, 
        route='aluva_pettah', 
        weather_factor=0.9  # Light rain
    )
    
    print(f"\n1. ENSEMBLE RIDERSHIP PREDICTION")
    print(f"   Date/Time: {test_datetime.strftime('%Y-%m-%d %H:%M')} (Friday, Peak Hour)")
    print(f"   Route: Aluva-Pettah")
    print(f"   Weather: Light rain (factor: 0.9)")
    print(f"   ")
    print(f"   Ensemble Prediction: {prediction_result['ensemble_prediction']} passengers")
    print(f"   Confidence Score: {prediction_result['confidence_score']:.2f}")
    print(f"   Individual Model Predictions:")
    for model, pred in prediction_result['individual_predictions'].items():
        print(f"     - {model.replace('_', ' ').title()}: {pred:.0f}")
    
    # 2. Daily passenger insights
    print(f"\n2. DAILY PASSENGER ANALYTICS")
    
    # Simulate some daily data
    test_date = "2025-09-15"
    
    # Add some sample daily records
    stations = ['Aluva', 'Edappally', 'Maharajas', 'Petta']
    for station in stations:
        entered = np.random.randint(800, 1500)
        exited = np.random.randint(700, 1400)
        ai_system.passenger_tracker.add_daily_record(
            date=test_date,
            station=station,
            entered=entered,
            exited=exited,
            route='aluva_pettah',
            weather_data={'temperature': 28, 'rainfall': 5}
        )
    
    daily_insights = ai_system.get_daily_passenger_insights(test_date)
    
    print(f"   Date: {test_date}")
    print(f"   Total Passengers: {daily_insights['passenger_metrics']['total_passengers']:,}")
    print(f"   Passengers Entered: {daily_insights['passenger_metrics']['passengers_entered']:,}")
    print(f"   Passengers Exited: {daily_insights['passenger_metrics']['passengers_exited']:,}")
    print(f"   Net Flow: {daily_insights['passenger_metrics']['net_passenger_flow']:+,}")
    print(f"   Busiest Station: {daily_insights['operational_metrics']['busiest_station']}")
    print(f"   Active Stations: {daily_insights['operational_metrics']['active_stations']}")
    
    if 'trend_analysis' in daily_insights:
        trend = daily_insights['trend_analysis']
        print(f"   Trend Analysis ({trend['station']}):")
        print(f"     - Direction: {trend['trend_direction']}")
        print(f"     - Change: {trend['trend_percentage']}")
        print(f"     - Recent Average: {trend['recent_average']}")
    
    # 3. Weekly forecast
    print(f"\n3. WEEKLY PASSENGER FORECAST")
    
    # Add historical data for better forecasting
    for day_offset in range(-7, 0):
        hist_date = (datetime.datetime.strptime(test_date, '%Y-%m-%d') + datetime.timedelta(days=day_offset)).strftime('%Y-%m-%d')
        for station in stations:
            base_passengers = np.random.randint(900, 1300)
            ai_system.passenger_tracker.add_daily_record(
                date=hist_date,
                station=station,
                entered=base_passengers // 2,
                exited=base_passengers // 2,
                route='aluva_pettah'
            )
    
    weekly_forecast = ai_system.predict_next_week_passengers(
        station='Aluva',
        weather_factors=[1.0, 0.8, 0.9, 1.0, 1.1, 0.7, 0.9]  # Varied weather
    )
    
    print(f"   Station: {weekly_forecast['station']}")
    print(f"   Total Week Forecast: {weekly_forecast['total_week_forecast']:,} passengers")
    print(f"   Daily Breakdown:")
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for pred in weekly_forecast['predictions']:
        day_name = days[pred['day'] - 1]
        print(f"     Day {pred['day']} ({day_name}): {pred['predicted_passengers']:,} passengers "
              f"(Confidence: {pred['confidence']:.1f})")
    
    # 4. Model comparison and performance
    print(f"\n4. MODEL PERFORMANCE COMPARISON")
    
    # Test all models on same data
    test_features = [
        [8, 4, 0.9, 1, 0, 9, 2, 1],  # Friday morning peak, light rain
        [14, 1, 1.0, 0, 0, 9, 2, 1],  # Tuesday afternoon, clear
        [19, 5, 1.0, 1, 1, 9, 2, 1],  # Saturday evening peak
    ]
    
    scenarios = [
        "Friday 8:00 AM (Peak, Light Rain)",
        "Tuesday 2:00 PM (Normal, Clear)", 
        "Saturday 7:00 PM (Weekend Peak)"
    ]
    
    print("   Scenario Predictions:")
    for i, (features, scenario) in enumerate(zip(test_features, scenarios)):
        print(f"\n   {scenario}:")
        
        try:
            nn_pred = ai_system.models['neural_network'].predict([features])[0]
            print(f"     Neural Network: {nn_pred:.0f} passengers")
        except:
            print(f"     Neural Network: Failed")
        
        try:
            rf_pred = ai_system.models['random_forest'].predict([features])[0]
            print(f"     Random Forest: {rf_pred:.0f} passengers")
        except:
            print(f"     Random Forest: Failed")
        
        try:
            gb_pred = ai_system.models['gradient_boosting'].predict([features])[0]
            print(f"     Gradient Boosting: {gb_pred:.0f} passengers")
        except:
            print(f"     Gradient Boosting: Failed")
    
    # 5. Station trend analysis
    print(f"\n5. STATION TREND ANALYSIS")
    
    for station in ['Aluva', 'Edappally']:
        trend = ai_system.passenger_tracker.get_station_trend(station, days=7)
        if trend:
            print(f"   {station} Station:")
            print(f"     - Trend: {trend['trend_direction']} ({trend['trend_percentage']:+.1f}%)")
            print(f"     - Recent Average: {trend['recent_average']:.0f} passengers/day")
            print(f"     - Daily Pattern: {trend['daily_totals']}")
        else:
            print(f"   {station} Station: Insufficient data for trend analysis")
    
    # 6. System capabilities summary
    print(f"\n6. SYSTEM CAPABILITIES SUMMARY")
    print(f"   ✓ Multiple AI Models: Neural Network, Random Forest, Gradient Boosting")
    print(f"   ✓ Ensemble Predictions with confidence scoring")
    print(f"   ✓ Day-to-day passenger flow tracking")
    print(f"   ✓ Station-wise trend analysis")
    print(f"   ✓ Weekly passenger forecasting")
    print(f"   ✓ Weather impact modeling")
    print(f"   ✓ Peak/off-peak hour optimization")
    print(f"   ✓ Route-specific predictions")
    print(f"   ✓ Real-time daily insights")
    
    print(f"\n[SUCCESS] KMRL Advanced AI System demonstration completed!")
    print(f"[INFO] The system is now ready for production use with your real dataset.")
    
    return ai_system


# Integration with your pandas dataset
def integrate_with_pandas_data():
    """Integration function for your specific pandas dataset"""
    print("\n*** Integration with Your Pandas Dataset ***")
    print("=" * 50)
    
    # Your original pandas data creation
    stations = ["Aluva", "Edappally", "Maharajas", "Petta"]
    dates = pd.date_range("2025-09-01", periods=30, freq='D')

    data = []
    for date in dates:
        for station in stations:
            entered = np.random.randint(100, 2000)
            exited = np.random.randint(80, 1900)
            fare = entered * np.random.uniform(20, 40)
            temp = np.random.uniform(24, 36)
            rain = np.random.choice([0, np.random.uniform(0, 30)], p=[0.7, 0.3])
            weekday = date.weekday()
            data.append([date, station, entered, exited, fare, temp, rain, weekday])

    df = pd.DataFrame(data, columns=["Date", "Station", "Passengers_Entered", "Passengers_Exited", "Fare_Collected", "Temperature", "Rainfall", "Weekday"])
    
    print(f"[DATA] Created pandas DataFrame with {len(df)} records")
    print(f"[DATA] Columns: {list(df.columns)}")
    
    # Initialize AI system
    ai_system = KMRLAdvancedAI()
    
    # Convert pandas data to AI training format
    training_data = []
    
    for _, row in df.iterrows():
        # Convert daily station data to hourly route data
        total_passengers = row['Passengers_Entered'] + row['Passengers_Exited']
        
        # Generate hourly distribution
        for hour in range(5, 24):
            # Peak hour multipliers
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                hour_multiplier = 2.0 / 19  # Peak hours get more passengers
            else:
                hour_multiplier = 0.5 / 19  # Off-peak hours get fewer
            
            hourly_passengers = int(total_passengers * hour_multiplier)
            
            # Weather factor calculation
            weather_factor = 1.0
            if row['Rainfall'] > 10:
                weather_factor *= 0.8
            if row['Temperature'] > 35:
                weather_factor *= 0.9
            
            training_record = {
                'date': row['Date'].strftime('%Y-%m-%d'),
                'hour': hour,
                'day_of_week': row['Weekday'],
                'route': 'aluva_pettah',  # All stations are on this route
                'ridership': hourly_passengers,
                'delay_minutes': max(0, np.random.normal(1 + (hourly_passengers / 100), 1)),
                'weather_factor': weather_factor,
                'is_peak': int(7 <= hour <= 9 or 17 <= hour <= 19),
                'is_weekend': int(row['Weekday'] >= 5),
                'frequency_minutes': 6,
                'month': row['Date'].month,
                'season': ai_system._get_season(row['Date'].month)
            }
            
            training_data.append(training_record)
            
            # Add to daily passenger tracker
            ai_system.passenger_tracker.add_daily_record(
                date=training_record['date'],
                station=row['Station'],
                entered=row['Passengers_Entered'],
                exited=row['Passengers_Exited'],
                route='aluva_pettah',
                weather_data={
                    'temperature': row['Temperature'],
                    'rainfall': row['Rainfall'],
                    'factor': weather_factor
                }
            )
    
    print(f"[PROCESSING] Converted to {len(training_data)} AI training records")
    
    # Train models
    ai_system.train_all_models(training_data)
    
    # Generate insights from your data
    print(f"\n[INSIGHTS] Analysis of Your Dataset:")
    
    # Daily summaries for recent dates
    recent_dates = df['Date'].dt.strftime('%Y-%m-%d').unique()[-5:]
    
    for date in recent_dates:
        insights = ai_system.get_daily_passenger_insights(date)
        if isinstance(insights, dict):
            print(f"\n  {date}:")
            print(f"    Total Passengers: {insights['passenger_metrics']['total_passengers']:,}")
            print(f"    Busiest Station: {insights['operational_metrics']['busiest_station']}")
    
    # Station trends
    print(f"\n[TRENDS] Station Performance Trends:")
    for station in stations:
        trend = ai_system.passenger_tracker.get_station_trend(station, days=10)
        if trend:
            print(f"  {station}: {trend['trend_direction']} ({trend['trend_percentage']:+.1f}%)")
    
    # Predictions for tomorrow
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    tomorrow_pred = ai_system.predict_ridership_ensemble(
        date_time=tomorrow.replace(hour=8, minute=0),
        route='aluva_pettah',
        weather_factor=0.95
    )
    
    print(f"\n[PREDICTION] Tomorrow's 8:00 AM Ridership Forecast:")
    print(f"  Ensemble Prediction: {tomorrow_pred['ensemble_prediction']} passengers")
    print(f"  Confidence: {tomorrow_pred['confidence_score']:.2f}")
    
    print(f"\n[SUCCESS] Your pandas dataset has been successfully integrated!")
    
    return ai_system, df, training_data


if __name__ == "__main__":
    # Run main demonstration
    ai_system = main()
    
    print(f"\n" + "="*60)
    print(f"To integrate with your pandas dataset, run:")
    print(f"ai_system, df, training_data = integrate_with_pandas_data()")
    
    # Uncomment the line below to run pandas integration
    # integrate_with_pandas_data()