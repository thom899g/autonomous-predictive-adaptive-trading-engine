"""
Hypothesis Generator (Cortex) - Core 1 of APATE Trinity Architecture
Generates testable market hypotheses using regime detection and anomaly identification.
Implements Gaussian Mixture Models for regime detection and statistical arbitrage signals.
"""

import structlog
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import ta

logger = structlog.get_logger(__name__)


@dataclass
class MarketHypothesis:
    """Data class representing a generated market hypothesis"""
    hypothesis_id: str
    statement: str
    confidence_score: float
    time_horizon: str  # 'short', 'medium', 'long'
    assets_involved: List[str]
    generated_at: datetime
    expected_validation_window: timedelta
    supporting_metrics: Dict[str, float]


class HypothesisGenerator:
    """Core Cortex component that generates testable market hypotheses"""
    
    def __init__(self, firestore_client):
        """
        Initialize Hypothesis Generator with Firebase integration for state persistence.
        
        Args:
            firestore_client: Initialized Firestore client for state management
        """
        self.firestore = firestore_client
        self.logger = structlog.get_logger(__name__)
        self.regime_model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.logger.info("HypothesisGenerator initialized", firebase_connected=firestore_client is not None)
    
    def detect_market_regime(self, market_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect current market regime using Gaussian Mixture Models.
        
        Args:
            market_data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            Dict containing regime classification and confidence metrics
            
        Raises:
            ValueError: If market_data is empty or insufficient for analysis
        """
        if market_data.empty or len(market_data) < 100:
            raise ValueError("Insufficient data for regime detection. Need at least 100 data points.")
        
        try:
            # Prepare features for regime detection
            features = self._extract_regime_features(market_data)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train or update GMM (online learning approach)
            if self.regime_model is None:
                self.regime_model = GaussianMixture(n_components=3, random_state=42)
                self.regime_model.fit(features_scaled)
            else:
                # Partial fit for online learning
                self.regime_model.fit(features_scaled)
            
            # Predict current regime
            current_regime = self.regime_model.predict(features_scaled[-1].reshape(1, -1))[0]
            regime_probs = self.regime_model.predict_proba(features_scaled[-1].reshape(1, -1))[0]
            
            regime_map = {0: "trending_bull", 1: "trending_bear", 2: "ranging"}
            regime_name = regime_map.get(current_regime, "unknown")
            
            self.logger.info(
                "Market regime detected",
                regime=regime_name,
                confidence=max(regime_probs),
                timestamp=market_data.index[-1]
            )
            
            return {
                "regime": regime_name,
                "confidence": float(max(regime_probs)),
                "probabilities": regime_probs.tolist(),
                "features_used": features.columns.tolist()
            }
            
        except Exception as e:
            self.logger.error("Regime detection failed", error=str(e), exc_info=True)
            raise
    
    def generate_liquidity_hypothesis(self, market_data: pd.DataFrame) -> Optional[MarketHypothesis]:
        """
        Generate hypothesis about liquidity flows and volatility spillover.
        
        Args:
            market_data: DataFrame containing order book and volume data
            
        Returns:
            MarketHypothesis or None if insufficient confidence
        """
        try:
            if market_data.empty:
                self.logger.warning("Empty market data for hypothesis generation")
                return None
            
            # Calculate liquidity metrics
            bid_ask_spread = self._calculate_bid_ask_spread(market_data)
            order_book_imbalance = self._calculate_order_book_imbalance(market_data)
            volume_profile = self._analyze_volume_profile(market_data)
            
            # Detect anomalies
            spread_anomaly = bid_ask_spread > (market_data['spread'].mean() + 2 * market_data['spread'].std())
            imbalance_anomaly = abs(order_book_imbalance) > 0.7
            
            if spread_anomaly and imbalance_anomaly:
                # Generate specific hypothesis
                direction = "draining from" if order_book_imbalance < -0.5 else "accumulating in"
                assets = ["BTC", "ETH"]  # Default, should be parameterized
                
                hypothesis = MarketHypothesis(
                    hypothesis_id=f"liq_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    statement=f"Liquidity is {direction} perpetual swaps, volatility spillover to spot likely in <2 hours",
                    confidence_score=0.75,
                    time_horizon="short",
                    assets_involved=assets,
                    generated_at=datetime.utcnow(),
                    expected_validation_window=timedelta(hours=2),
                    supporting_metrics={
                        "bid_ask_spread": float(bid_ask_spread),
                        "order_book_imbalance": float(order_book_imbalance),
                        "volume_concentration": float(volume_profile['concentration']),
                        "spread_zscore": float