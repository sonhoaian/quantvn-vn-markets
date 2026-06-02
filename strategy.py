from quantvn import client

import pandas as pd
import numpy as np

from quantvn.vn.data import get_stock_hist

from dotenv import load_dotenv
import os

# Load các biến môi trường từ file .env
load_dotenv()

# Lấy giá trị biến APS
aps_value = os.getenv('MY_API_KEY')


client(apikey=aps_value)


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA + MACD + BB là 3 chỉ báo chính.

    Nếu cả 3 đồng thuận:
        => quyết định ngay

    Nếu không đồng thuận:
        => RSI làm trọng tài

    Output:
        position:
            1  = BUY
            0  = HOLD
           -1  = SELL
    """

    df = df.copy()

    # ==================================================
    # EMA
    # ==================================================

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # Độ dốc EMA20 trong 5 phiên

    df["EMA20_SLOPE"] = (
        df["EMA20"]
        - df["EMA20"].shift(5)
    )

    ema_signal = np.where(
        (df["EMA20"] > df["EMA50"])
        &
        (df["EMA20_SLOPE"] > 0),

        1,

        np.where(
            (df["EMA20"] < df["EMA50"])
            &
            (df["EMA20_SLOPE"] < 0),

            -1,

            0
        )
    )

    # ==================================================
    # MACD
    # ==================================================

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    macd_diff = (
        df["MACD"]
        - df["MACD_SIGNAL"]
    )

    # Threshold giúp giảm nhiễu

    threshold = 0.05

    macd_signal = np.where(
        macd_diff > threshold,

        1,

        np.where(
            macd_diff < -threshold,

            -1,

            0
        )
    )

    # ==================================================
    # RSI
    # ==================================================

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = (
        100
        - (100 / (1 + rs))
    )

    # ==================================================
    # BOLLINGER BAND
    # ==================================================

    bb_mid = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["Close"]
        .rolling(20)
        .std()
    )

    df["BB_UPPER"] = bb_mid + 2 * bb_std
    df["BB_LOWER"] = bb_mid - 2 * bb_std

    df["BB_WIDTH"] = (
        df["BB_UPPER"]
        - df["BB_LOWER"]
    ) / bb_mid

    bb_width_change = (
        df["BB_WIDTH"]
        - df["BB_WIDTH"].shift(5)
    )

    bb_signal = np.where(
        (df["Close"] > bb_mid)
        &
        (bb_width_change > 0),

        1,

        np.where(
            (df["Close"] < bb_mid)
            &
            (bb_width_change > 0),

            -1,

            0
        )
    )

    # ==================================================
    # GHI LẠI TỪNG PHIẾU ĐỂ DEBUG
    # ==================================================

    df["EMA_SIGNAL"] = ema_signal
    df["MACD_SIGNAL_VOTE"] = macd_signal
    df["BB_SIGNAL"] = bb_signal

    # ==================================================
    # QUY TẮC ĐỒNG THUẬN TUYỆT ĐỐI
    # ==================================================

    df["position"] = 0

    buy_consensus = (
        (ema_signal == 1)
        &
        (macd_signal == 1)
        &
        (bb_signal == 1)
    )

    sell_consensus = (
        (ema_signal == -1)
        &
        (macd_signal == -1)
        &
        (bb_signal == -1)
    )

    hold_consensus = (
        (ema_signal == 0)
        &
        (macd_signal == 0)
        &
        (bb_signal == 0)
    )

    # --------------------------------------------------
    # Đồng thuận tuyệt đối
    # --------------------------------------------------

    df.loc[
        buy_consensus,
        "position"
    ] = 1

    df.loc[
        sell_consensus,
        "position"
    ] = -1

    df.loc[
        hold_consensus,
        "position"
    ] = 0

    # ==================================================
    # RSI TRỌNG TÀI
    # ==================================================

    conflict = ~(
        buy_consensus
        |
        sell_consensus
        |
        hold_consensus
    )

    # RSI nghiêng về phe mua

    df.loc[
        conflict
        &
        (df["RSI"] > 60),

        "position"
    ] = 1

    # RSI nghiêng về phe bán

    df.loc[
        conflict
        &
        (df["RSI"] < 40),

        "position"
    ] = -1

    # RSI trung lập

    df.loc[
        conflict
        &
        (df["RSI"] >= 40)
        &
        (df["RSI"] <= 60),

        "position"
    ] = 0

    return df


# ==================================================
# TEST
# ==================================================

df = get_stock_hist(
    "FPT",
    "15m"
)

df_pos = gen_position(df)

print(
    df_pos[
        [
            "Date",
            "Close",
            "EMA_SIGNAL",
            "MACD_SIGNAL_VOTE",
            "BB_SIGNAL",
            "RSI",
            "position"
        ]
    ].tail(20)
)