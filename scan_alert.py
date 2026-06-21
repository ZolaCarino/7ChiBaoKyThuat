import os
import requests
import pandas as pd
import pandas_ta as ta
from vnstock.ui import Market
from datetime import datetime, timedelta

# DANH SÁCH CỔ PHIẾU ANH MUỐN THEO DÕI (Có thể tự do thêm/bớt)
WATCHLIST = ['SSI', 'PVD', 'PVS', 'HPG', 'DIG', 'VND', 'TCB', 'STB', 'MWG']

# Cấu hình Telegram lấy từ biến môi trường hệ thống (Bảo mật)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_html(text):
    """Hàm gửi tin nhắn định dạng HTML về Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

def check_stock_score(symbol):
    """Thuật toán gốc của anh - Trả về điểm số và hành động"""
    end_date = str(datetime.now().date())
    start_date = str((datetime.now() - timedelta(days=365)).date())
    try:
        mkt = Market()
        df = mkt.equity(symbol).ohlcv(start=start_date, end=end_date)
        if df is None or df.empty: return None
        
        df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 1. Tính toán các chỉ báo bằng pandas-ta
        bbands = df.ta.bbands(length=20, std=2)
        rsi = df.ta.rsi(length=14)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        psar = df.ta.psar(af0=0.02, af=0.02, max_af=0.2)
        ichi_df, _ = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52)
        
        if bbands is None or rsi is None or macd is None or psar is None or ichi_df is None:
            return None
            
        df = pd.concat([df, bbands, rsi, macd, psar, ichi_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        
        def safe_val(col_target, offset=-1):
            res = df[col_target]
            if isinstance(res, pd.DataFrame): res = res.iloc[:, 0]
            return float(res.iloc[offset])
        
        close_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        volume_series = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        high_series = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low_series = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']

        # --- VPVR ---
        num_bins = 30
        p_min, p_max = float(low_series.min()), float(high_series.max())
        if p_max > p_min:
            bins = pd.cut(close_series, bins=num_bins)
            vp = df.groupby(bins, observed=False)['Volume'].sum()
            poc_price = float(vp.idxmax().mid)
        else:
            poc_price = float(close_series.iloc[-1])
        poc_top, poc_bottom = poc_price * 1.005, poc_price * 0.995

        # Dò cột chỉ báo tự động
        bbl_col = [c for c in df.columns if c.upper().startswith('BBL')][0]
        bbm_col = [c for c in df.columns if c.upper().startswith('BBM')][0]
        bbu_col = [c for c in df.columns if c.upper().startswith('BBU')][0]
        rsi_col = [c for c in df.columns if c.upper().startswith('RSI')][0]
        macd_col = [c for c in df.columns if c.upper().startswith('MACD_') and not c.upper().endswith('H') and not c.upper().endswith('S')][0]
        macds_col = [c for c in df.columns if c.upper().startswith('MACDS_') or c.upper().startswith('MACD') and c.upper().endswith('S')][0]
        tenkan_col = [c for c in df.columns if c.upper().startswith('ITS_')][0]
        kijun_col = [c for c in df.columns if c.upper().startswith('IKS_')][0]
        isa_col = [c for c in df.columns if c.upper().startswith('ISA_')][0]
        isb_col = [c for c in df.columns if c.upper().startswith('ISB_')][0]
        
        close_curr, close_prev = safe_val('Close', -1), safe_val('Close', -2)
        low_curr, high_curr = safe_val('Low', -1), safe_val('High', -1)
        bbl_curr, bbm_curr, bbm_prev, bbu_curr = safe_val(bbl_col, -1), safe_val(bbm_col, -1), safe_val(bbm_col, -2), safe_val(bbu_col, -1)
        rsi_curr, rsi_prev = safe_val(rsi_col, -1), safe_val(rsi_col, -2)
        macd_curr, macd_prev, macds_curr, macds_prev = safe_val(macd_col, -1), safe_val(macd_col, -2), safe_val(macds_col, -1), safe_val(macds_col, -2)
        tenkan_curr, tenkan_prev, kijun_curr, kijun_prev = safe_val(tenkan_col, -1), safe_val(tenkan_col, -2), safe_val(kijun_col, -1), safe_val(kijun_col, -2)
        isa_curr, isb_curr = safe_val(isa_col, -1), safe_val(isb_col, -1)

        # BB Score
        bb_score = 0
        if close_prev <= bbm_prev and close_curr > bbm_curr: bb_score = 1
        elif low_curr <= bbl_curr and close_curr > bbl_curr: bb_score = 1
        elif close_prev >= bbm_prev and close_curr < bbm_curr: bb_score = -1
        elif high_curr >= bbu_curr and close_curr < bbu_curr: bb_score = -1

        # RSI Score
        rsi_score = 0
        if rsi_curr <= 30 or (rsi_prev < 50 and rsi_curr >= 50): rsi_score = 1
        elif rsi_curr >= 70 or (rsi_prev > 50 and rsi_curr <= 50): rsi_score = -1

        # MACD Score
        macd_score = 0
        if macd_prev <= macds_prev and macd_curr > macds_curr: macd_score += 1
        elif macd_prev >= macds_prev and macd_curr < macds_curr: macd_score -= 1
        
        # SAR Score
        sar_score = 0
        if pd.notna(df[[c for c in df.columns if c.upper().startswith('PSARL')][0]].iloc[-1]): sar_score = 1
        else: sar_score = -1

        # Ichimoku
        ichi_score = 0
        kumo_top, kumo_bottom = max(isa_curr, isb_curr), min(isa_curr, isb_curr)
        if close_curr > kumo_top and tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr: ich_score = 1
        elif close_curr < kumo_bottom and tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr: ich_score = -1

        # VPVR
        vpvr_score = 0
        if close_curr > poc_top: vpvr_score = 1
        elif close_curr < poc_bottom: vpvr_score = -1

        # Vol Kicker
        vol_ma20 = float(volume_series.rolling(window=20).mean().iloc[-1])
        vol_curr = safe_val('Volume', -1)
        base_score = bb_score + rsi_score + macd_score + sar_score + ichi_score + vpvr_score
        vol_kicker = 0
        if vol_curr > 1.3 * vol_ma20:
            vol_kicker = 1 if base_score > 0 else (-1 if base_score < 0 else 0)

        total_score = base_score + vol_kicker
        return {"score": total_score, "price": close_curr}
    except Exception as e:
        print(f"Lỗi phân tích mã {symbol}: {e}")
        return None

def main():
    now_str = datetime.now().strftime('%Y-%m-%d')
    # Bắt đầu xây dựng chuỗi tin nhắn HTML gửi Telegram
    msg = f"🔔 <b>HỆ THỐNG QUÉT TÍN HIỆU CUỐI PHIÊN ({now_str})</b>\n"
    msg += f"<i>Tiêu chí: Chỉ hiển thị mã đạt Ưu thế Mua (Từ 2 đến 9 điểm)</i>\n"
    msg += "=========================\n"
    
    has_signal = False
    
    for symbol in WATCHLIST:
        res = check_stock_score(symbol)
        if res and res["score"] >= 2: # Lọc chỉ lấy các mã có điểm số Tích cực trở lên
            has_signal = True
            score = res["score"]
            price = res["price"]
            
            status = "🟩 MUA MẠNH" if score >= 6 else "🟨 MUA THĂM DÒ"
            msg += f"• <b>{symbol}</b> | Giá: <code>{price:,}</code>\n"
            msg += f"  👉 Trạng thái: <b>{status}</b> (<code>{score} điểm</code>)\n\n"
            
    if not has_signal:
        msg += "Không có mã nào đạt điều kiện mua trong phiên hôm nay."
        
    msg += "=========================\n"
    msg += "📊 <i>Hệ thống phân tích tự động bởi GitHub Bot</i>"
    
    send_telegram_html(msg)

if __name__ == "__main__":
    main()
