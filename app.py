import streamlit as st
import pandas as pd
import pandas_ta as ta
from vnstock.ui import Market
from datetime import datetime, timedelta

st.set_page_config(page_title="Hệ Thống Chấm Điểm Kỹ Thuật", page_icon="📊", layout="centered")

st.title("📊 Hệ Thống Tra Cứu Tín Hiệu Cổ Phiếu")
st.write("Nhập mã cổ phiếu để hệ thống tự động quét dữ liệu thời gian thực và chấm điểm kỹ thuật.")

# Ô nhập mã cổ phiếu trên giao diện Web
symbol = st.text_input("Nhập mã cổ phiếu (Ví dụ: SSI, PVD, HPG):", value="SSI").strip().upper()

def check_stock_score(symbol):
    end_date = str(datetime.now().date())
    start_date = str((datetime.now() - timedelta(days=365)).date())
    try:
        mkt = Market()
        df = mkt.equity(symbol).ohlcv(start=start_date, end=end_date)
        if df is None or df.empty: return None
        
        df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Tính toán các chỉ báo bằng pandas_ta
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
        high_series = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low_series = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume_series = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

        # Thuật toán tính VPVR
        num_bins = 30
        p_min, p_max = float(low_series.min()), float(high_series.max())
        if p_max > p_min:
            bins = pd.cut(close_series, bins=num_bins)
            vp = df.groupby(bins, observed=False)['Volume'].sum()
            poc_price = float(vp.idxmax().mid)
        else:
            poc_price = float(close_series.iloc[-1])
        poc_top, poc_bottom = poc_price * 1.005, poc_price * 0.995

        # Dò chính xác tên cột của thư viện pandas_ta
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

        # Logic chấm điểm hệ thống chỉ báo
        bb_score = 1 if (close_prev <= bbm_prev and close_curr > bbm_curr) or (low_curr <= bbl_curr and close_curr > bbl_curr) else (-1 if (close_prev >= bbm_prev and close_curr < bbm_curr) or (high_curr >= bbu_curr and close_curr < bbu_curr) else 0)
        rsi_score = 1 if rsi_curr <= 30 or (rsi_prev < 50 and rsi_curr >= 50) else (-1 if rsi_curr >= 70 or (rsi_prev > 50 and rsi_curr <= 50) else 0)
        macd_score = 1 if macd_prev <= macds_prev and macd_curr > macds_curr else -1 if macd_prev >= macds_prev and macd_curr < macds_curr else 0
        sar_score = 1 if pd.notna(df[[c for c in df.columns if c.upper().startswith('PSARL')][0]].iloc[-1]) else -1
        ichi_score = 1 if close_curr > max(isa_curr, isb_curr) and tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr else -1 if close_curr < min(isa_curr, isb_curr) and tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr else 0
        vpvr_score = 1 if close_curr > poc_top else (-1 if close_curr < poc_bottom else 0)

        # Tác vụ tính Vol Kicker
        vol_ma20 = float(volume_series.rolling(window=20).mean().iloc[-1])
        vol_curr = safe_val('Volume', -1)
        base_score = bb_score + rsi_score + macd_score + sar_score + ichi_score + vpvr_score
        vol_kicker = 1 if (vol_curr > 1.3 * vol_ma20 and base_score > 0) else (-1 if (vol_curr > 1.3 * vol_ma20 and base_score < 0) else 0)

        return {"score": base_score + vol_kicker, "price": close_curr}
    except Exception as e:
        st.error(f"Lỗi phân tích: {e}")
        return None

# Thực thi hiển thị kết quả ra màn hình Web
if symbol:
    if len(symbol) == 3 and symbol.isalpha():
        with st.spinner(f"Đang quét dữ liệu mã {symbol}..."):
            res = check_stock_score(symbol)
            if res:
                score = res["score"]
                price = res["price"]
                
                if score >= 6: status, color = "🟩 MUA MẠNH", "green"
                elif score >= 2: status, color = "🟨 MUA THĂM DÒ", "orange"
                elif score <= -6: status, color = "🟥 BÁN MẠNH", "red"
                elif score <= -2: status, color = "🟧 BÁN HẠ TỶ TRỌNG", "red"
                else: status, color = "🟪 THEO DÕI (Trung lập)", "normal"
                
                st.subheader(f"📊 Kết quả phân tích mã: {symbol}")
                st.metric(label="Giá hiện tại", value=f"{price:,} VNĐ")
                st.metric(label="Tổng điểm kỹ thuật", value=f"{score} / 7 điểm")
                st.markdown(f"👉 Khuyến nghị hành động: **{status}**")
            else:
                st.warning("Không lấy được dữ liệu. Anh kiểm tra lại mã cổ phiếu nhé.")
    else:
        st.error("Mã không hợp lệ. Vui lòng nhập đúng 3 ký tự chữ.")
