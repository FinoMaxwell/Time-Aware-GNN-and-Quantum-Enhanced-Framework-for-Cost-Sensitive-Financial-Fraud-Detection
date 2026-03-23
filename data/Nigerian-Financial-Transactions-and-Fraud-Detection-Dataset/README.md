---
license: gpl
language:
- en
tags:
- finance
size_categories:
- 1M<n<10M
---
Nigerian Financial Fraud Detection Dataset (Enhanced)

## Overview

This is a comprehensive synthetic financial fraud detection dataset specifically engineered for the Nigerian fintech ecosystem. The dataset contains **5,000,000 transactions** with **45 advanced features** including sophisticated user behaviour analytics, device intelligence, risk scoring, and temporal patterns tailored for Nigerian financial fraud detection.

## We have found that a lot of people are unable to use the full dataset because of compute constraints. If you are splitting the dataset, make sure you specify the ratio of fraud to non-fraud signals within the amount that you collect. The sweet spot is 27%. If you do not do this, your script might select unintelligently, and you might not get enough fraud signals to get a great model. 

### Key Highlights

- 🇳🇬 **Nigerian-Localized**: Currency (NGN), cities, payment channels, IP ranges
- 🧠 **Advanced Features**: 30 sophisticated fraud detection features
- 📊 **Production-Ready**: No data leakage, optimized for ML training
- 🔍 **Behavioral Analytics**: User personas, transaction patterns, device intelligence
- ⚡ **High-Performance**: Engineered for 5M+ transaction processing

## Dataset Statistics

- **Total Transactions**: 5,000,000
- **Total Features**: 45
- **Fraud Rate**: ~15% (realistic for emerging markets)
- **Time Span**: Simulated 12-month period
- **Unique Users**: ~500,000 sender accounts
- **Nigerian Cities**: 20 major cities across 6 geo-regions

## Feature Categories

### 🏦 Core Transaction Features (15 features)

- `transaction_id`: Unique transaction identifier
- `sender_account`: Anonymized 10-digit Nigerian account number
- `receiver_account`: Anonymized 10-digit Nigerian account number
- `amount_ngn`: Transaction amount in Nigerian Naira (NGN)
- `timestamp`: Transaction timestamp with Nigerian business patterns
- `payment_channel`: Nigerian payment methods (USSD, Mobile App, Card, Bank Transfer)
- `merchant_category`: Localized merchants (Jumia, MTN Airtime, Bet9ja, etc.)
- `location`: Major Nigerian cities
- `ip_address`: IP addresses from Nigerian IP ranges
- `device_hash`: Anonymized device identifier
- `is_fraud`: Binary fraud label (0=legitimate, 1=fraud)
- `fraud_type`: Specific fraud categories (Account Takeover, Identity Fraud, etc.)
- `bvn_linked`: Bank Verification Number linkage status
- `sender_persona`: Behavioral profile (Salary Earner, Student, Trader)
- `geospatial_velocity_anomaly`: Impossible travel pattern flag

### 👤 User Behavior Features (5 features)

- `user_avg_txn_amt`: Average transaction amount per user (expanding window)
- `user_std_txn_amt`: Standard deviation of user's transaction amounts
- `user_top_category`: Most frequent merchant category per user
- `user_txn_frequency_24h`: Transaction frequency indicator
- `user_txn_count_total`: Lifetime transaction count per user

### 📱 Device & IP Intelligence (5 features)

- `device_seen_count`: Total transactions from each device
- `is_device_shared`: Flag for devices used by multiple users (0/1)
- `ip_seen_count`: Total transactions from each IP address
- `is_ip_shared`: Flag for IPs used by multiple users (0/1)
- `ip_geo_region`: Nigerian geo-region (South West, North Central, etc.)

### 📈 Transaction History/Window Features (4 features)

- `txn_count_last_1h`: Recent transaction count (expanding window)
- `txn_count_last_24h`: 24-hour transaction count (expanding window)
- `total_amount_last_1h`: Sum of recent transactions (expanding window)
- `avg_gap_between_txns`: Average time between user transactions (minutes)

### ⚠️ Risk Scoring Fields (4 features)

- `merchant_fraud_rate`: Historical fraud rate per merchant category (no leakage)
- `channel_risk_score`: Risk score by payment channel (USSD=0.8, Mobile=0.6, etc.)
- `persona_fraud_risk`: Risk by user persona (Trader=0.7, Student=0.5, etc.)
- `location_fraud_risk`: Historical fraud rate per Nigerian city (no leakage)

### 🕐 Temporal Features (4 features)

- `txn_hour`: Transaction hour (0-23)
- `is_weekend`: Weekend transaction flag (0/1)
- `is_salary_week`: Last 5 days of month flag (0/1)
- `is_night_txn`: Night transaction flag 11pm-5am (0/1)

### 🔧 Technical Features (8 features)

- `new_device_for_sender`: First-time device usage flag
- `shared_device_hash`: Device used by multiple accounts
- `time_since_last`: Minutes since user's last transaction
- Various derived flags and indicators

## Nigerian Localization Details

### 🏙️ Geographic Coverage

**6 Nigerian Geo-Regions Represented:**

- **South West**: Lagos, Ibadan, Abeokuta, Oyo, Ogbomoso
- **North Central**: Abuja, Jos, Ilorin, Okene
- **North West**: Kano, Zaria, Kaduna, Sokoto
- **South South**: Port Harcourt, Benin City, Warri
- **South East**: Aba, Enugu, Onitsha
- **North East**: Maiduguri

### 💳 Payment Channels (Risk-Weighted)

- **USSD** (Risk: 0.8) - Most common, highest fraud risk
- **Mobile App** (Risk: 0.6) - Moderate risk
- **Card** (Risk: 0.4) - Lower risk
- **Bank Transfer** (Risk: 0.3) - Lowest risk

### 🛒 Merchant Categories (Nigerian Context)

- Jumia Purchase, Konga Shopping, MTN Airtime Top-up
- Airtel Data Bundle, Bet9ja Stake, NairaBet Gaming
- Uber Ride, Bolt Transport, Flutterwave Payment
- Paystack Transaction, Opay Transfer, PalmPay Service
- And 10+ other Nigerian-specific merchants

### 👥 User Personas (Behavioral Profiles)

- **Salary Earner** (Risk: 0.4): Regular monthly patterns, moderate amounts
- **Student** (Risk: 0.5): Small frequent transactions, education-related
- **Trader** (Risk: 0.7): High-volume, irregular patterns, higher risk

## Fraud Types and Scenarios

### 🚨 Fraud Categories Included

1. **Account Takeover**: Compromised accounts with unusual patterns
2. **Identity Fraud**: Fake accounts, unlinked BVN
3. **Impossible Travel Fraud**: Geospatial velocity anomalies
4. **SIM Swap Fraud**: Device changes with suspicious activity
5. **Card-Not-Present**: Online fraud without physical card
6. **Deposit Fraud**: Fake deposit schemes
7. **Money Laundering**: Structured transactions, unusual patterns

### 🎯 Fraud Detection Signals

- BVN linkage status (unlinked = higher risk)
- New device usage patterns
- Shared device/IP indicators
- Geospatial velocity anomalies
- Off-hours transaction patterns
- Unusual merchant category combinations
- Rapid transaction sequences

## Technical Implementation

### 🔒 Data Privacy & Security

- All account numbers are randomly generated (no real accounts)
- Device hashes are anonymized identifiers
- IP addresses are from public Nigerian ranges
- No personally identifiable information (PII) included

### 📊 Feature Engineering Methodology

- **No Data Leakage**: Risk scores use expanding windows with `.shift(1)`
- **Temporal Consistency**: Features computed chronologically
- **Performance Optimized**: Efficient groupby operations for 5M+ rows
- **Realistic Patterns**: Based on Nigerian fintech transaction behaviors

### 🔄 Rolling Window Logic

```python
# Example: Merchant fraud rate without leakage
merchant_fraud_rate = df.groupby('merchant_category')['is_fraud']
                       .transform(lambda x: x.expanding().mean().shift(1).fillna(0.1))
```

## Use Cases & Applications

### 🎯 Primary Use Cases

1. **Fraud Detection Model Training**: Supervised learning with rich feature set
2. **Risk Scoring System Development**: Real-time transaction scoring
3. **Behavioral Analytics**: User pattern analysis and segmentation
4. **Anomaly Detection**: Unsupervised fraud detection research
5. **Nigerian Fintech Research**: Emerging market fraud patterns

### 🏢 Target Industries

- Nigerian fintech companies (Flutterwave, Paystack, Opay, etc.)
- Traditional banks expanding digital services
- Fraud detection technology vendors
- Academic research institutions
- Regulatory bodies and compliance teams

### 📈 Model Performance Expectations

- **Baseline Accuracy**: 85-90% with basic features
- **Enhanced Accuracy**: 92-95% with advanced features
- **Precision/Recall**: Optimized for Nigerian fraud patterns
- **Real-time Scoring**: Features designed for low-latency inference

## Data Quality & Validation

### ✅ Quality Assurance

- **Completeness**: No missing values in critical features
- **Consistency**: Logical relationships maintained across features
- **Realism**: Patterns based on Nigerian financial behaviors
- **Balance**: Appropriate fraud/legitimate transaction ratio

### 🔍 Validation Metrics

- Transaction amounts follow realistic Nigerian distributions
- Temporal patterns align with Nigerian business hours
- Geographic distribution matches Nigerian population centers
- Fraud patterns consistent with emerging market trends

## Limitations & Considerations

### ⚠️ Important Limitations

1. **Synthetic Data**: Not real transactions, patterns may differ from reality
2. **Temporal Scope**: Limited to 12-month simulation period
3. **Feature Completeness**: May not capture all real-world fraud signals
4. **Regional Focus**: Specific to Nigerian context, may not generalize
5. **Fraud Evolution**: Real fraud patterns evolve faster than synthetic data

### 🎯 Recommended Usage

- Use as training data supplement, not replacement for real data
- Validate model performance on real Nigerian transaction data
- Regular model retraining as fraud patterns evolve
- Combine with external data sources for production systems

## Technical Specifications

### 📋 File Details

- **Format**: CSV (Comma-separated values)
- **Size**: ~2.5GB uncompressed
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Header**: Yes (first row contains column names)

### 🔧 System Requirements

- **Memory**: 8GB+ RAM recommended for full dataset loading
- **Storage**: 5GB+ available space
- **Processing**: Multi-core CPU recommended for feature engineering
- **Software**: Python 3.7+, pandas 1.3+, numpy 1.20+

## Getting Started

### 📚 Quick Start Example

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('financial_fraud_detection_dataset_nigeria.csv')

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
print(f"Features: {df.columns.tolist()}")

# Feature selection for modeling
feature_cols = [col for col in df.columns if col not in ['transaction_id', 'is_fraud', 'fraud_type']]
X = df[feature_cols]
y = df['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2%}")
```

### 🎯 Advanced Analytics Examples

```python
# Analyze fraud patterns by region
region_fraud = df.groupby('ip_geo_region')['is_fraud'].agg(['count', 'mean'])
print("Fraud rates by Nigerian region:")
print(region_fraud)

# User behavior analysis
user_stats = df.groupby('sender_persona').agg({
    'amount_ngn': ['mean', 'std'],
    'is_fraud': 'mean',
    'user_txn_count_total': 'mean'
})
print("\nUser persona analysis:")
print(user_stats)

# Temporal fraud patterns
temporal_fraud = df.groupby(['txn_hour', 'is_weekend'])['is_fraud'].mean()
print("\nTemporal fraud patterns:")
print(temporal_fraud)
```

## Citation & Attribution

### 📄 Recommended Citation

```
Nigerian Financial Fraud Detection Dataset (Enhanced)
Synthetic dataset for fraud detection research in Nigerian fintech
Generated: 2024
Features: 45 advanced fraud detection features
Transactions: 5,000,000 synthetic transactions
```

### 🏷️ Tags

`fraud-detection` `nigeria` `fintech` `machine-learning` `synthetic-data` `behavioral-analytics` `risk-scoring` `emerging-markets` `financial-crime` `anomaly-detection`

## Support & Updates

### 📞 Contact Information

For questions, issues, or collaboration opportunities regarding this dataset, please refer to the project documentation or contact the development team.

### 🔄 Version History

- **v2.0** (Current): Enhanced with 30 advanced features, Nigerian localization
- **v1.0**: Basic fraud detection dataset with core features

### 🚀 Future Enhancements

- Transaction network analysis features
- Additional Nigerian payment channels
- Seasonal fraud pattern variations
- Enhanced merchant category coverage
- Real-time streaming data simulation
