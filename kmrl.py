import numpy as np
import pandas as pd
import datetime
import math
import random
from collections import defaultdict

class KMRLAIScheduler:
    def __init__(self):
        self.ridership_model = None
        self.delay_model = None
        self.optimization_model = None
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
        
        # Peak hours for Kochi
        self.peak_hours = {
            'morning': (7, 10),
            'evening': (17, 20),
            'night': (21, 23)
        }
        
        # ML Models (Custom implementations)
        self.patterns = defaultdict(list)
        self.weights = {}
        
    def generate_synthetic_data(self, days=365):
        """Generate synthetic historical data for training"""
        random.seed(42)
        np.random.seed(42) if 'numpy' in globals() else None
        data = []
        
        for day in range(days):
            date = datetime.datetime.now() - datetime.timedelta(days=day)
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday
            
            for hour in range(5, 24):  # Operating hours 5 AM to 11 PM
                for route in self.routes.keys():
                    # Base ridership patterns
                    base_ridership = self._calculate_base_ridership(hour, day_of_week, route)
                    
                    # Add weather and special event factors
                    weather_factor = random.gauss(1, 0.1)
                    special_event_factor = 1.3 if random.random() < 0.05 else 1.0
                    
                    ridership = int(max(0, base_ridership * weather_factor * special_event_factor))
                    
                    # Calculate delays based on ridership and other factors
                    delay = self._calculate_delay(ridership, hour, day_of_week)
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'route': route,
                        'ridership': ridership,
                        'delay_minutes': delay,
                        'weather_factor': weather_factor,
                        'is_peak': self._is_peak_hour(hour),
                        'is_weekend': day_of_week >= 5,
                        'frequency_minutes': self._optimal_frequency(ridership, hour)
                    })
        
        return data
    
    def _calculate_base_ridership(self, hour, day_of_week, route):
        """Calculate base ridership based on historical patterns"""
        route_multiplier = 1.5 if route == 'aluva_pettah' else 0.8
        
        # Hour-based pattern
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
            base = 350 * route_multiplier
        elif 10 <= hour <= 16:  # Daytime
            base = 180 * route_multiplier
        elif 20 <= hour <= 22:  # Evening
            base = 220 * route_multiplier
        else:  # Early morning/late night
            base = 80 * route_multiplier
            
        # Day of week adjustment
        if day_of_week >= 5:  # Weekend
            base *= 0.7
            
        return base + random.gauss(0, 30)
    
    def _calculate_delay(self, ridership, hour, day_of_week):
        """Calculate expected delay based on ridership and conditions"""
        base_delay = 0.5
        
        # Ridership impact
        if ridership > 300:
            base_delay += (ridership - 300) * 0.01
            
        # Peak hour impact
        if self._is_peak_hour(hour):
            base_delay += 1.5
            
        # Random variation
        delay = base_delay + random.expovariate(2)  # Exponential distribution
        
        return min(delay, 15)  # Cap at 15 minutes
    
    def _is_peak_hour(self, hour):
        """Check if given hour is peak hour"""
        return (7 <= hour <= 9) or (17 <= hour <= 19)
    
    def _optimal_frequency(self, ridership, hour):
        """Calculate optimal train frequency based on ridership"""
        if ridership > 400:
            return 3  # High frequency
        elif ridership > 200:
            return 5  # Medium frequency
        elif self._is_peak_hour(hour):
            return 4  # Peak hour frequency
        else:
            return 8  # Low frequency

    class SimpleLinearRegression:
        """Custom implementation of Linear Regression"""
        def __init__(self):
            self.weights = None
            self.bias = None
            
        def fit(self, X, y):
            # Add bias column
            X = [[1] + row for row in X]
            X_matrix = self._matrix_multiply(self._transpose(X), X)
            X_y = self._matrix_vector_multiply(self._transpose(X), y)
            
            # Solve using normal equation: (X^T * X)^(-1) * X^T * y
            try:
                X_inv = self._matrix_inverse(X_matrix)
                weights = self._matrix_vector_multiply(X_inv, X_y)
                self.bias = weights[0]
                self.weights = weights[1:]
            except:
                # Fallback to simple average if matrix inversion fails
                self.weights = [0.1] * (len(X[0]) - 1)
                self.bias = sum(y) / len(y)
        
        def predict(self, X):
            if self.weights is None:
                return [0] * len(X)
            
            predictions = []
            for row in X:
                pred = self.bias + sum(w * x for w, x in zip(self.weights, row))
                predictions.append(pred)
            return predictions
        
        def _transpose(self, matrix):
            return list(map(list, zip(*matrix)))
        
        def _matrix_multiply(self, A, B):
            result = []
            for i in range(len(A)):
                row = []
                for j in range(len(B[0])):
                    row.append(sum(A[i][k] * B[k][j] for k in range(len(B))))
                result.append(row)
            return result
        
        def _matrix_vector_multiply(self, A, v):
            return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]
        
        def _matrix_inverse(self, matrix):
            n = len(matrix)
            # Create augmented matrix
            aug = [row[:] + [0] * n for row in matrix]
            for i in range(n):
                aug[i][n + i] = 1
            
            # Gaussian elimination
            for i in range(n):
                # Make diagonal 1
                pivot = aug[i][i]
                if abs(pivot) < 1e-10:
                    raise ValueError("Matrix is singular")
                
                for j in range(2 * n):
                    aug[i][j] /= pivot
                
                # Make column 0
                for k in range(n):
                    if k != i:
                        factor = aug[k][i]
                        for j in range(2 * n):
                            aug[k][j] -= factor * aug[i][j]
            
            # Extract inverse
            return [row[n:] for row in aug]

    class SimpleDecisionTree:
        """Custom implementation of Decision Tree for regression"""
        def __init__(self, max_depth=5):
            self.max_depth = max_depth
            self.tree = None
            
        def fit(self, X, y):
            self.tree = self._build_tree(X, y, 0)
        
        def predict(self, X):
            return [self._predict_single(row, self.tree) for row in X]
        
        def _build_tree(self, X, y, depth):
            if depth >= self.max_depth or len(set(y)) == 1:
                return sum(y) / len(y)  # Return average
            
            best_feature = 0
            best_threshold = 0
            best_gain = -1
            
            # Find best split
            for feature in range(len(X[0])):
                values = [row[feature] for row in X]
                thresholds = set(values)
                
                for threshold in thresholds:
                    gain = self._calculate_gain(X, y, feature, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
            
            if best_gain == -1:
                return sum(y) / len(y)
            
            # Split data
            left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)
            
            return {
                'feature': best_feature,
                'threshold': best_threshold,
                'left': self._build_tree(left_X, left_y, depth + 1),
                'right': self._build_tree(right_X, right_y, depth + 1)
            }
        
        def _calculate_gain(self, X, y, feature, threshold):
            left_y = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
            right_y = [y[i] for i in range(len(X)) if X[i][feature] > threshold]
            
            if len(left_y) == 0 or len(right_y) == 0:
                return -1
            
            # Calculate variance reduction
            original_var = self._variance(y)
            left_var = self._variance(left_y)
            right_var = self._variance(right_y)
            
            weighted_var = (len(left_y) * left_var + len(right_y) * right_var) / len(y)
            return original_var - weighted_var
        
        def _variance(self, y):
            if len(y) == 0:
                return 0
            mean = sum(y) / len(y)
            return sum((val - mean) ** 2 for val in y) / len(y)
        
        def _split_data(self, X, y, feature, threshold):
            left_X, left_y, right_X, right_y = [], [], [], []
            
            for i in range(len(X)):
                if X[i][feature] <= threshold:
                    left_X.append(X[i])
                    left_y.append(y[i])
                else:
                    right_X.append(X[i])
                    right_y.append(y[i])
            
            return left_X, left_y, right_X, right_y
        
        def _predict_single(self, row, node):
            if isinstance(node, (int, float)):
                return node
            
            if row[node['feature']] <= node['threshold']:
                return self._predict_single(row, node['left'])
            else:
                return self._predict_single(row, node['right'])

    def train_models(self, data):
        """Train ML models for ridership prediction and optimization"""
        print("Training AI/ML models...")
        
        self.training_data = data
        
        # Prepare features
        X_ridership = []
        y_ridership = []
        X_delay = []
        y_delay = []
        X_freq = []
        y_freq = []
        
        for record in data:
            features = [
                record['hour'],
                record['day_of_week'],
                record['weather_factor'],
                int(record['is_peak']),
                int(record['is_weekend']),
                1 if record['route'] == 'aluva_pettah' else 0
            ]
            
            X_ridership.append(features)
            y_ridership.append(record['ridership'])
            
            X_delay.append(features)
            y_delay.append(record['delay_minutes'])
            
            X_freq.append(features + [record['ridership']])
            y_freq.append(record['frequency_minutes'])
        
        # Train models
        self.ridership_model = self.SimpleDecisionTree(max_depth=8)
        self.ridership_model.fit(X_ridership, y_ridership)
        
        self.delay_model = self.SimpleDecisionTree(max_depth=6)
        self.delay_model.fit(X_delay, y_delay)
        
        self.optimization_model = self.SimpleLinearRegression()
        self.optimization_model.fit(X_freq, y_freq)
        
        # Evaluate models (simple accuracy check)
        ridership_pred = self.ridership_model.predict(X_ridership[:100])
        ridership_mae = sum(abs(p - a) for p, a in zip(ridership_pred, y_ridership[:100])) / 100
        print(f"Ridership Prediction MAE: {ridership_mae:.2f}")
        
        delay_pred = self.delay_model.predict(X_delay[:100])
        delay_mae = sum(abs(p - a) for p, a in zip(delay_pred, y_delay[:100])) / 100
        print(f"Delay Prediction MAE: {delay_mae:.2f} minutes")
        
        freq_pred = self.optimization_model.predict(X_freq[:100])
        freq_mae = sum(abs(p - a) for p, a in zip(freq_pred, y_freq[:100])) / 100
        print(f"Frequency Optimization MAE: {freq_mae:.2f} minutes")
        
        print("Model training completed!")
    
    def predict_ridership(self, datetime_obj, route, weather_factor=1.0):
        """Predict ridership for given conditions"""
        if self.ridership_model is None:
            raise Exception("Models not trained. Call train_models() first.")
        
        features = [
            datetime_obj.hour,
            datetime_obj.weekday(),
            weather_factor,
            int(self._is_peak_hour(datetime_obj.hour)),
            int(datetime_obj.weekday() >= 5),
            1 if route == 'aluva_pettah' else 0
        ]
        
        predicted_ridership = self.ridership_model.predict([features])[0]
        return max(0, int(predicted_ridership))
    
    def predict_delay(self, datetime_obj, route, weather_factor=1.0):
        """Predict delay for given conditions"""
        if self.delay_model is None:
            raise Exception("Models not trained. Call train_models() first.")
        
        features = [
            datetime_obj.hour,
            datetime_obj.weekday(),
            weather_factor,
            int(self._is_peak_hour(datetime_obj.hour)),
            int(datetime_obj.weekday() >= 5),
            1 if route == 'aluva_pettah' else 0
        ]
        
        predicted_delay = self.delay_model.predict([features])[0]
        return max(0, predicted_delay)
    
    def optimize_schedule(self, date_str, route):
        """Generate optimized schedule for a full day"""
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        schedule = []
        
        # Generate schedule from 5 AM to 11 PM
        for hour in range(5, 24):
            for minute in [0, 30]:  # Check every 30 minutes
                current_time = date_obj.replace(hour=hour, minute=minute)
                
                # Predict ridership and delay
                predicted_ridership = self.predict_ridership(current_time, route)
                predicted_delay = self.predict_delay(current_time, route)
                
                # Calculate optimal frequency
                optimal_freq = self._calculate_optimal_frequency(predicted_ridership, current_time)
                
                # Calculate efficiency score
                efficiency = self._calculate_efficiency(predicted_ridership, optimal_freq)
                
                schedule.append({
                    'time': current_time.strftime('%H:%M'),
                    'predicted_ridership': predicted_ridership,
                    'predicted_delay': round(predicted_delay, 2),
                    'optimal_frequency_min': optimal_freq,
                    'efficiency_score': round(efficiency, 2),
                    'recommendation': self._get_recommendation(predicted_ridership, predicted_delay, efficiency)
                })
        
        return schedule
    
    def _calculate_optimal_frequency(self, ridership, datetime_obj):
        """Calculate optimal train frequency using ML model"""
        if self.optimization_model is None:
            # Fallback logic
            if ridership > 400:
                return 3
            elif ridership > 250:
                return 4
            elif ridership > 150:
                return 6
            else:
                return 8
        
        features = [
            datetime_obj.hour,
            datetime_obj.weekday(),
            1.0,  # weather_factor
            int(self._is_peak_hour(datetime_obj.hour)),
            int(datetime_obj.weekday() >= 5),
            1,    # route_encoded (aluva_pettah)
            ridership
        ]
        
        optimal_freq = self.optimization_model.predict([features])[0]
        return max(3, min(10, int(optimal_freq)))
    
    def _calculate_efficiency(self, ridership, frequency):
        """Calculate efficiency score"""
        # Trains per hour
        trains_per_hour = 60 / frequency
        
        # Assume train capacity of 300
        capacity_per_hour = trains_per_hour * 300
        
        # Efficiency as utilization percentage
        efficiency = (ridership / capacity_per_hour) * 100 if capacity_per_hour > 0 else 0
        
        return min(100, efficiency)
    
    def _get_recommendation(self, ridership, delay, efficiency):
        """Generate AI recommendation"""
        if delay > 5:
            return "HIGH_PRIORITY: Increase frequency to reduce delays"
        elif efficiency > 90:
            return "OPTIMAL: Current schedule is efficient"
        elif efficiency < 50:
            return "REDUCE_FREQUENCY: Low ridership detected"
        elif ridership > 350:
            return "INCREASE_CAPACITY: Consider express services"
        else:
            return "MAINTAIN: Current schedule is adequate"
    
    def real_time_optimization(self, current_ridership, current_delay, next_hour_weather=1.0):
        """Real-time schedule adjustment based on current conditions"""
        now = datetime.datetime.now()
        
        adjustments = {
            'frequency_change': 0,
            'capacity_change': 0,
            'route_modification': None,
            'alert_level': 'NORMAL'
        }
        
        # Predict next hour ridership
        next_hour = now + datetime.timedelta(hours=1)
        predicted_ridership = self.predict_ridership(next_hour, 'aluva_pettah', next_hour_weather)
        
        # Real-time adjustments
        if current_delay > 8:
            adjustments['frequency_change'] = -2  # Increase frequency (reduce interval)
            adjustments['alert_level'] = 'HIGH'
        elif current_ridership > predicted_ridership * 1.5:
            adjustments['frequency_change'] = -1
            adjustments['capacity_change'] = 1
            adjustments['alert_level'] = 'MEDIUM'
        
        if predicted_ridership > 500:
            adjustments['route_modification'] = 'ADD_EXPRESS_SERVICE'
            
        return adjustments
    
    def generate_daily_insights(self, date_str, route):
        """Generate AI insights for the day"""
        schedule = self.optimize_schedule(date_str, route)
        
        insights = {
            'peak_ridership_time': None,
            'max_predicted_ridership': 0,
            'high_delay_periods': [],
            'efficiency_score': 0,
            'recommendations': []
        }
        
        # Analyze schedule
        for slot in schedule:
            if slot['predicted_ridership'] > insights['max_predicted_ridership']:
                insights['max_predicted_ridership'] = slot['predicted_ridership']
                insights['peak_ridership_time'] = slot['time']
            
            if slot['predicted_delay'] > 5:
                insights['high_delay_periods'].append(slot['time'])
        
        # Calculate average efficiency
        efficiency_scores = [slot['efficiency_score'] for slot in schedule]
        insights['efficiency_score'] = sum(efficiency_scores) / len(efficiency_scores)
        
        # Generate recommendations
        if insights['efficiency_score'] < 60:
            insights['recommendations'].append("Consider reducing off-peak frequency")
        
        if len(insights['high_delay_periods']) > 5:
            insights['recommendations'].append("Implement dynamic frequency adjustment")
            
        if insights['max_predicted_ridership'] > 450:
            insights['recommendations'].append("Deploy additional trains during peak hours")
        
        return insights

# Example usage and demonstration
def main():
    print("*** KMRL AI/ML Train Scheduling System ***")
    print("=" * 50)
    
    # Initialize the AI scheduler
    scheduler = KMRLAIScheduler()
    
    # Generate synthetic training data
    print("\n[DATA] Generating synthetic training data...")
    training_data = scheduler.generate_synthetic_data(days=365)
    print(f"Generated {len(training_data)} data points")
    
    # Train ML models
    scheduler.train_models(training_data)
    
    # Demonstrate predictions
    print("\n[AI] Predictions Demo:")
    test_date = datetime.datetime(2024, 3, 15, 8, 30)  # Friday morning peak
    
    ridership_pred = scheduler.predict_ridership(test_date, 'aluva_pettah')
    delay_pred = scheduler.predict_delay(test_date, 'aluva_pettah')
    
    print(f"Date/Time: {test_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Route: Aluva-Pettah")
    print(f"Predicted Ridership: {ridership_pred} passengers")
    print(f"Predicted Delay: {delay_pred:.2f} minutes")
    
    # Generate optimized schedule
    print("\n[SCHEDULE] Optimized Schedule for Today:")
    schedule = scheduler.optimize_schedule('2024-03-15', 'aluva_pettah')
    
    print("Time    | Ridership | Delay | Freq | Efficiency | Recommendation")
    print("-" * 80)
    
    for i, slot in enumerate(schedule):
        if i % 4 == 0:  # Show every 2 hours
            print(f"{slot['time']}  | {slot['predicted_ridership']:9d} | "
                  f"{slot['predicted_delay']:5.1f} | {slot['optimal_frequency_min']:4d} | "
                  f"{slot['efficiency_score']:9.1f}% | {slot['recommendation']}")
    
    # Real-time optimization demo
    print("\n[REAL-TIME] Optimization Demo:")
    current_ridership = 380  # Current high ridership
    current_delay = 6.5      # Current delay
    
    adjustments = scheduler.real_time_optimization(current_ridership, current_delay)
    
    print(f"Current Conditions: {current_ridership} riders, {current_delay} min delay")
    print(f"AI Recommendations:")
    print(f"  - Frequency Change: {adjustments['frequency_change']} minutes")
    print(f"  - Alert Level: {adjustments['alert_level']}")
    if adjustments['route_modification']:
        print(f"  - Route Modification: {adjustments['route_modification']}")
    
    # Daily insights
    print("\n[INSIGHTS] AI Analysis for Today:")
    insights = scheduler.generate_daily_insights('2024-03-15', 'aluva_pettah')
    
    print(f"Peak Ridership Time: {insights['peak_ridership_time']}")
    print(f"Max Predicted Ridership: {insights['max_predicted_ridership']} passengers")
    print(f"High Delay Periods: {len(insights['high_delay_periods'])} time slots")
    print(f"Overall Efficiency Score: {insights['efficiency_score']:.1f}%")
    print("Recommendations:")
    for rec in insights['recommendations']:
        print(f"  * {rec}")
    
    print("\n[SUCCESS] AI/ML Train Scheduling System Demo Complete!")

if __name__ == "__main__":
    main()