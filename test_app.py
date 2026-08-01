import unittest
import json
from app import app

class FraudSystemTestCase(unittest.TestCase):
    def setUp(self):
        # Configure the app for testing
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_homepage(self):
        """Test if the homepage renders correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Credit Card Fraud Detection', response.data)

    def test_legitimate_prediction(self):
        """Test a prediction that should represent a typical legitimate transaction."""
        payload = {
            "amt": 15.50,
            "category": "grocery_pos",
            "gender": "F",
            "state": "NY",
            "age": 34,
            "city_pop": 800000,
            "merchant": "fraud_Kozey-McGlynn"
        }
        
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['prediction'], 'legitimate')
        self.assertIn('probability', result)
        self.assertIn('confidence', result)
        self.assertLess(result['probability'], 0.5)

    def test_fraudulent_prediction(self):
        """Test a prediction that should represent a suspicious transaction (high amount)."""
        payload = {
            "amt": 850.00,  # suspicious amount
            "category": "shopping_net",
            "gender": "M",
            "state": "CA",
            "age": 45,
            "city_pop": 1500000,
            "merchant": "fraud_Kozey-McGlynn"
        }
        
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['prediction'], 'fraud')
        self.assertIn('probability', result)
        self.assertGreater(result['probability'], 0.5)

    def test_invalid_input_validation(self):
        """Test that negative transaction amounts fail validation."""
        payload = {
            "amt": -10.00,  # invalid negative amount
            "category": "travel",
            "gender": "F",
            "state": "TX",
            "age": 28,
            "city_pop": 50000,
            "merchant": "fraud_Kozey-McGlynn"
        }
        
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        result = json.loads(response.data)
        self.assertEqual(result['status'], 'error')
        self.assertIn('amount must be greater than 0', result['message'])

if __name__ == '__main__':
    unittest.main()
