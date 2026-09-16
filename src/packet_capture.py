import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠️ Scapy not installed. Install with: pip install scapy")

class PacketAnalyzer:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.packet_history = []
        self.alert_history = []
        
        # Feature mapping for extracted packet data
        self.feature_columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
    
    def packet_to_features(self, packet):
        """Convert a packet to NIDS feature vector"""
        features = [0] * 41  # Default values
        
        # Extract basic features
        if IP in packet:
            ip = packet[IP]
            features[0] = 0  # duration (default)
            
            # Protocol
            if TCP in packet:
                features[1] = 0  # tcp
                features[3] = 3  # SF flag
                features[4] = len(packet[TCP].payload)  # src_bytes
                features[5] = 0  # dst_bytes
            elif UDP in packet:
                features[1] = 1  # udp
                features[4] = len(packet[UDP].payload)
            elif ICMP in packet:
                features[1] = 2  # icmp
                features[4] = len(packet[ICMP].payload)
            
            # Service detection (simplified)
            if TCP in packet:
                if packet[TCP].dport == 80 or packet[TCP].sport == 80:
                    features[2] = 0  # http
                elif packet[TCP].dport == 443:
                    features[2] = 1  # https
                elif packet[TCP].dport == 22:
                    features[2] = 2  # ssh
                elif packet[TCP].dport == 21:
                    features[2] = 3  # ftp
                elif packet[TCP].dport == 25:
                    features[2] = 4  # smtp
                elif packet[TCP].dport == 53:
                    features[2] = 5  # dns
                else:
                    features[2] = 6  # other
        
        return features
    
    def analyze_packet(self, packet):
        """Analyze a single packet"""
        try:
            # Convert to features
            features = self.packet_to_features(packet)
            
            # Create DataFrame
            df = pd.DataFrame([features], columns=self.feature_columns)
            
            # Preprocess
            processed = self.preprocessor.prepare_prediction_data(df)
            
            # Predict
            prediction = self.model.predict(processed)[0]
            probability = self.model.predict_proba(processed)[0]
            
            # Store in history
            self.packet_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'probability': max(probability),
                'features': features
            })
            
            # Keep only last 100 packets
            if len(self.packet_history) > 100:
                self.packet_history = self.packet_history[-100:]
            
            return {
                'prediction': prediction,
                'probability': max(probability),
                'is_attack': prediction == 1
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def start_live_capture(self, count=10, timeout=30):
        """Start real-time packet capture"""
        if not SCAPY_AVAILABLE:
            print("❌ Scapy not available")
            return []
        
        print(f"\n📡 Starting packet capture...")
        print(f"📊 Capturing {count} packets or timeout {timeout}s")
        
        def packet_callback(pkt):
            result = self.analyze_packet(pkt)
            if result:
                print(f"  {'🚨 ATTACK' if result.get('is_attack') else '✅ Normal'}"
                      f" - Confidence: {result.get('probability', 0):.2%}")
                self.alert_history.append(result)
        
        # Start capture
        sniff(prn=packet_callback, count=count, timeout=timeout)
        
        print(f"\n✅ Capture complete! Analyzed {len(self.alert_history)} packets")
        return self.alert_history

if __name__ == "__main__":
    print("📡 Packet Capture Module Ready")
    print("Use from main application")