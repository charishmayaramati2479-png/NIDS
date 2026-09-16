import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os

class AlertSystem:
    def __init__(self, config_file='config/alerts.json'):
        self.config = self.load_config(config_file)
        self.alert_history = []
    
    def load_config(self, config_file):
        """Load alert configuration"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender': '',
                'password': '',
                'recipients': []
            },
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': ''
            },
            'webhook': {
                'enabled': False,
                'url': ''
            }
        }
        
        try:
            os.makedirs('config', exist_ok=True)
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                    return config
            else:
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except:
            return default_config
    
    def send_email_alert(self, subject, message):
        """Send email alert"""
        if not self.config['email']['enabled']:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['sender']
            msg['To'] = ', '.join(self.config['email']['recipients'])
            msg['Subject'] = f"[NIDS ALERT] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            )
            server.starttls()
            server.login(
                self.config['email']['sender'],
                self.config['email']['password']
            )
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email alert sent: {subject}")
            return True
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
            return False
    
    def send_telegram_alert(self, message):
        """Send Telegram alert"""
        if not self.config['telegram']['enabled']:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
            payload = {
                'chat_id': self.config['telegram']['chat_id'],
                'text': f"🚨 NIDS ALERT\n\n{message}",
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("✅ Telegram alert sent")
            return True
        except Exception as e:
            print(f"❌ Telegram alert failed: {e}")
            return False
    
    def send_webhook_alert(self, data):
        """Send webhook alert"""
        if not self.config['webhook']['enabled']:
            return False
        
        try:
            response = requests.post(
                self.config['webhook']['url'],
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            print("✅ Webhook alert sent")
            return True
        except Exception as e:
            print(f"❌ Webhook alert failed: {e}")
            return False
    
    def send_alert(self, prediction_data):
        """Send alert through all enabled channels"""
        if not prediction_data.get('is_attack', False):
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"""
🚨 INTRUSION DETECTED

Time: {timestamp}
Confidence: {prediction_data.get('probability', 0):.2%}
Type: Attack Detected

Features: {prediction_data.get('features', '')}
        """
        
        # Send through all enabled channels
        self.send_email_alert("Intrusion Detected!", message)
        self.send_telegram_alert(message)
        self.send_webhook_alert(prediction_data)
        
        # Log to history
        self.alert_history.append({
            'timestamp': timestamp,
            'message': message,
            'data': prediction_data
        })
        
        return True

if __name__ == "__main__":
    alert = AlertSystem()
    print("✅ Alert system initialized")
    print("Configure alerts in config/alerts.json")