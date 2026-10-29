from .base import DecisionModel, Prediction


class MarketPriceModel(DecisionModel):
    """Baseline: trust the crowd. Predicts the current market price."""

    name = "market"
    uses_market_price = True

    def predict(self, market):
        return Prediction(market.yes_price, "Current Polymarket price")


class CoinFlipModel(DecisionModel):
    """Baseline: always 50%. Any useful model should beat this."""

    name = "coinflip"

    def predict(self, market):
        return Prediction(0.5, "Always 50/50")
