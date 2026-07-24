import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
from vnstock.ui import Market
from datetime import datetime, timedelta
import concurrent.futures
import os

WATCHLIST_FILE = "watchlist.txt"
DEFAULT_WATCHLIST = ["SSI", "HPG", "VCI", "PVT", "FPT", "FRT", "MWG", "ACB", "TCB"]

def load_watchlist():
    """Hàm đọc rổ cổ phiếu từ file, nếu file chưa có thì lấy rổ mặc định"""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                tickers = [line.strip().upper() for line in f.readlines() if line.strip()]
                if tickers:
                    return tickers
        except Exception:
            pass
    return DEFAULT_WATCHLIST

def save_watchlist(watchlist):
    """Hàm ghi đè rổ cổ phiếu hiện tại vào file lưu trữ"""
    try:
        with open(WATCHLIST_FILE, "w") as f:
            for ticker in watchlist:
                f.write(f"{ticker}\n")
    except Exception:
        pass
        
# Thiết lập cấu hình trang hiển thị chuẩn Mobile-First
st.set_page_config(
    page_title="Hệ Thống Phân Tích Kỹ Thuật Cổ Phiếu",
    page_icon="📊",
    layout="centered"
)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data_mac(symbol, show_error=True):
    """Kéo dữ liệu 1 năm bằng Unified UI"""
    end_date = str(datetime.now().date())
    start_date = str((datetime.now() - timedelta(days=365)).date())
    
    try:
        mkt = Market()
        df = mkt.equity(symbol).ohlcv(start=start_date, end=end_date)
        
        if df is None or df.empty:
            if show_error:
                st.error(f"[-] Không tìm thấy dữ liệu hoặc mã {symbol} không tồn tại.")
            return None
        
        df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        if show_error:
            st.error(f"[-] Lỗi kết nối hệ thống với mã {symbol}: {e}")
        return None
        
def fetch_data_pure(symbol, start_date, end_date):
    """Hàm tải dữ liệu thuần túy, an toàn tuyệt đối khi chạy đa luồng (Không gọi st.*)"""
    try:
        from vnstock.ui import Market
        mkt = Market()
        df = mkt.equity(symbol).ohlcv(start=start_date, end=end_date)
        
        if df is None or df.empty:
            return None
            
        df = df.copy()
        df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df
    except Exception:
        return None

def compute_signals(df):
    """Hàm lõi chuyên tính toán toán học và trả về bộ chỉ báo + điểm số kỹ thuật"""
    try:
        # 1. Tính toán các chỉ báo bằng pandas-ta-classic
        bbands = df.ta.bbands(length=20, std=2)
        rsi = df.ta.rsi(length=14)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        psar = df.ta.psar(af0=0.02, af=0.02, max_af=0.2)
        ichi_df = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52)
        
        if bbands is None or rsi is None or macd is None or psar is None or ichi_df is None:
            return None
            
        df = pd.concat([df, bbands, rsi, macd, psar, ichi_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        
        def safe_val(col_target, offset=-1):
            res = df[col_target]
            if isinstance(res, pd.DataFrame):
                res = res.iloc[:, 0]
            return float(res.iloc[offset])
        
        close_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        volume_series = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        high_series = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low_series = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']

        # --- THUẬT TOÁN VPVR ---
        num_bins = 30
        p_min = float(low_series.min())
        p_max = float(high_series.max())
        
        if p_max > p_min:
            bins = pd.cut(close_series, bins=num_bins)
            vp = df.groupby(bins, observed=False)['Volume'].sum()
            poc_bin = vp.idxmax()
            poc_price = float(poc_bin.mid)
        else:
            poc_price = float(close_series.iloc[-1])

        poc_top = poc_price * 1.005
        poc_bottom = poc_price * 0.995

        # Dò cột động
        bbl_col = [c for c in df.columns if c.upper().startswith('BBL')][0]
        bbm_col = [c for c in df.columns if c.upper().startswith('BBM')][0]
        bbu_col = [c for c in df.columns if c.upper().startswith('BBU')][0]
        rsi_col = [c for c in df.columns if c.upper().startswith('RSI')][0]
        
        macd_col = [c for c in df.columns if c.upper().startswith('MACD_') and not c.upper().endswith('H') and not c.upper().endswith('S')][0]
        macds_col = [c for c in df.columns if c.upper().startswith('MACDS_') or c.upper().startswith('MACD') and c.upper().endswith('S')][0]
        
        psarl_col = [c for c in df.columns if c.upper().startswith('PSARL')][0]
        psars_col = [c for c in df.columns if c.upper().startswith('PSARS')][0]
        
        tenkan_col = [c for c in df.columns if c.upper().startswith('ITS_')][0]
        kijun_col = [c for c in df.columns if c.upper().startswith('IKS_')][0]
        isa_col = [c for c in df.columns if c.upper().startswith('ISA_')][0]
        isb_col = [c for c in df.columns if c.upper().startswith('ISB_')][0]
        
        # Giá trị hiện tại
        close_curr = safe_val('Close', -1)
        close_prev = safe_val('Close', -2)
        low_curr = safe_val('Low', -1)
        high_curr = safe_val('High', -1)
        
        bbl_curr = safe_val(bbl_col, -1)
        bbm_curr = safe_val(bbm_col, -1)
        bbm_prev = safe_val(bbm_col, -2)
        bbu_curr = safe_val(bbu_col, -1)
        
        # Tính toán thông số phục vụ Bollinger Bands nâng cấp
        bw_series = (df[bbu_col] - df[bbl_col]) / df[bbm_col]
        bw_curr = float(bw_series.iloc[-1])
        bw_prev = float(bw_series.iloc[-2])
        bw_min_20 = float(bw_series.iloc[-20:].min())
        
        pct_b_curr = (close_curr - bbl_curr) / (bbu_curr - bbl_curr) if (bbu_curr - bbl_curr) != 0 else 0.5
        
        # --- ĐÁNH GIÁ MỨC ĐỘ ĐÓNG / MỞ CỦA BANDWIDTH ---
        bw_change_ratio = (bw_curr / bw_prev) if bw_prev != 0 else 1.0
        if bw_curr <= 1.10 * bw_min_20:
            bw_desc_status = "Đang thắt chặt rất mạnh (Squeeze - Tích lũy cực nén)"
        elif bw_change_ratio >= 1.15:
            bw_desc_status = "Đang mở dải mạnh (Bùng nổ biến động)"
        elif bw_change_ratio > 1.03:
            bw_desc_status = "Đang chớm mở rộng (Biến động gia tăng)"
        elif bw_change_ratio < 0.97:
            bw_desc_status = "Đang co hẹp dần (Thu hẹp biến động)"
        else:
            bw_desc_status = "Đang đi ngang ổn định (Biến động trung bình)"
        
        vol_ma20_series = volume_series.rolling(window=20).mean()
        vol_curr = safe_val('Volume', -1)
        vol_ma20_curr = float(vol_ma20_series.iloc[-1])

        rsi_curr = safe_val(rsi_col, -1)
        rsi_prev = safe_val(rsi_col, -2)
        
        macd_curr = safe_val(macd_col, -1)
        macd_prev = safe_val(macd_col, -2)
        macds_curr = safe_val(macds_col, -1)
        macds_prev = safe_val(macds_col, -2)

        tenkan_curr = safe_val(tenkan_col, -1)
        tenkan_prev = safe_val(tenkan_col, -2)
        kijun_curr = safe_val(kijun_col, -1)
        kijun_prev = safe_val(kijun_col, -2)
        isa_curr = safe_val(isa_col, -1)
        isb_curr = safe_val(isb_col, -1)

        # ==========================================
        # BOLLINGER BANDS RULE MATRIX
        # ==========================================
        bb_score = 0.0
        bb_status = "Trung tính (SideWay / Theo dõi)"
        bb_desc = "Giá biến động an toàn trong dải, chưa xuất hiện tín hiệu thắt chặt hay mở dải rõ rệt."

        # ƯU TIÊN 1: Bùng nổ Siêu Tốc (Breakout)
        if (bw_curr > 1.15 * bw_prev) and (pct_b_curr >= 0.85) and (vol_curr >= 1.5 * vol_ma20_curr):
            bb_score = 2.0
            bb_status = "Bùng nổ dòng tiền (Breakout 🚀)"
            bb_desc = f"Giá áp sát / bám sát biên trên (%B = {pct_b_curr*100:.1f}%), dải BB mở rộng mạnh (+{((bw_curr/bw_prev)-1)*100:.1f}%) kết hợp Volume bùng nổ ({vol_curr/vol_ma20_curr:.1f}x MA20) -> Xác nhận sóng tăng Siêu Tốc."

        # ƯU TIÊN 2: Mở dải Bán tháo (Walking Lower)
        elif (bw_curr > 1.15 * bw_prev) and (pct_b_curr <= 0.15) and (vol_curr >= 1.2 * vol_ma20_curr):
            bb_score = -2.0
            bb_status = "Mở dải giảm mạnh (Walking Lower 🔴)"
            bb_desc = f"Giá cắm sâu tiệm cận biên dưới (%B = {pct_b_curr*100:.1f}%), dải BB mở rộng xuống dưới đi kèm Volume bán gia tăng -> Tín hiệu xả hàng / mở rộng đà rơi."

        # ƯU TIÊN 3: Tích lũy Siết dải (Squeeze)
        elif (bw_curr <= 1.10 * bw_min_20) and (0.40 <= pct_b_curr <= 0.60):
            bb_score = 1.0
            bb_status = "Tích lũy chặt (Squeeze 🎯)"
            bb_desc = f"Dải BB siết chặt ở mức thấp nhất 20 phiên, giá dao động ổn định quanh trục giữa MA20 (%B = {pct_b_curr*100:.1f}%) -> Lực nén tích lũy rất mạnh, sẵn sàng chờ điểm nổ."

        # ƯU TIÊN 4: Sóng tăng bền vững
        elif (0.60 <= pct_b_curr < 0.85) and (bbm_curr > bbm_prev) and (vol_curr >= 1.0 * vol_ma20_curr):
            bb_score = 1.0
            bb_status = "Sóng tăng ổn định (Bullish 📈)"
            bb_desc = f"Giá duy trì thế dốc lên nằm trên MA20 (%B = {pct_b_curr*100:.1f}%), đường giữa MA20 hướng lên rõ nét và Volume ở mức khá -> Xu hướng tăng trưởng bền vững."

        # ƯU TIÊN 5: Suy yếu / Mất mốc MA20
        elif (pct_b_curr < 0.40) or (close_curr < bbm_curr):
            bb_score = -1.0
            bb_status = "Thủng hỗ trợ MA20 (Weak ⚠️)"
            bb_desc = f"Giá rơi xuống dưới đường giữa MA20 hoặc lùi sâu về sát biên dưới (%B = {pct_b_curr*100:.1f}%) -> Trạng thái kỹ thuật ngắn hạn bị suy yếu."

        bb_signal = bb_status
        bb_reason = bb_desc

        # --- RSI ---
        rsi_score = 0.0
        rsi_signal = "TRUNG TÍNH"
        rsi_reason = "RSI đang biến động ở vùng an toàn, chưa vi phạm quá mua/quá bán"
        if rsi_curr <= 30:
            rsi_score = 1.0
            rsi_signal = "QUÁ BÁN (Oversold)"
            rsi_reason = f"RSI lọt vào vùng QUÁ BÁN ({round(rsi_curr, 1)} <= 30) -> Áp lực bán suy kiệt"
        elif rsi_prev < 50 and rsi_curr >= 50:
            rsi_score = 1.0
            rsi_signal = "TÍCH CỰC (Bullish)"
            rsi_reason = "RSI cắt lên mốc trung vị 50 -> Phe Mua đang lấy lại thế trận"
        elif rsi_curr >= 70:
            rsi_score = -1.0
            rsi_signal = "QUÁ MUA (Overbought)"
            rsi_reason = f"RSI lọt vào vùng QUÁ MUA ({round(rsi_curr, 1)} >= 70) -> Lực mua quá tải"
        elif rsi_prev > 50 and rsi_curr <= 50:
            rsi_score = -1.0
            rsi_signal = "TIÊU CỰC (Bearish)"
            rsi_reason = "RSI cắt xuống mốc trung vị 50 -> Phe Bán đang kiểm soát thế trận"

        # --- MACD ---
        macd_score = 0.0
        macd_signal = "TRUNG TÍNH"
        macd_reasons = []
        if macd_prev <= macds_prev and macd_curr > macds_curr:
            macd_score += 1.0
            macd_signal = "MUA (Giao cắt Vàng)"
            macd_reasons.append("Đường MACD cắt lên trên đường Tín hiệu (Signal Line)")
        elif macd_prev >= macds_prev and macd_curr < macds_curr:
            macd_score -= 1.0
            macd_signal = "BÁN (Giao cắt Tử thần)"
            macd_reasons.append("Đường MACD cắt xuống dưới đường Tín hiệu (Signal Line)")

        last_trough_idx, last_peak_idx = None, None
        for i in range(len(df) - 4, len(df) - 35, -1):
            if i - 3 >= 0 and i + 4 <= len(df):
                if close_series.iloc[i] == min(close_series.iloc[i-3:i+4]):
                    last_trough_idx = i
                    break
        for i in range(len(df) - 4, len(df) - 35, -1):
            if i - 3 >= 0 and i + 4 <= len(df):
                if close_series.iloc[i] == max(close_series.iloc[i-3:i+4]):
                    last_peak_idx = i
                    break

        if last_trough_idx is not None:
            if close_curr < safe_val('Close', last_trough_idx) and macd_curr > safe_val(macd_col, last_trough_idx):
                macd_score += 0.5
                macd_signal = "MUA MẠNH (Phân kỳ dương 📈)"
                macd_reasons.append(f"PHÂN KỲ DƯƠNG: Giá phá đáy cũ ({df.index[last_trough_idx].strftime('%Y-%m-%d')}) nhưng MACD tạo đáy sau cao hơn.")
        if last_peak_idx is not None:
            if close_curr > safe_val('Close', last_peak_idx) and macd_curr < safe_val(macd_col, last_peak_idx):
                macd_score -= 0.5
                macd_signal = "BÁN MẠNH (Phân kỳ âm 📉)"
                macd_reasons.append(f"PHÂN KỲ ÂM: Giá vượt đỉnh cũ ({df.index[last_peak_idx].strftime('%Y-%m-%d')}) nhưng MACD tạo đỉnh sau thấp hơn.")
        if not macd_reasons:
            macd_reasons.append("Đường MACD và Tín hiệu đi ngang ổn định.")

        macd_score = max(-1.5, min(1.5, macd_score))

        # --- PARABOLIC SAR ---
        sar_score, sar_signal, sar_reason, sar_val = 0.0, "TRUNG TÍNH", "Không xác định rõ trạng thái.", 0.0
        psarl_series = df[psarl_col].iloc[:, 0] if isinstance(df[psarl_col], pd.DataFrame) else df[psarl_col]
        psars_series = df[psars_col].iloc[:, 0] if isinstance(df[psars_col], pd.DataFrame) else df[psars_col]

        if pd.notna(psarl_series.iloc[-1]):
            sar_val = psarl_series.iloc[-1]
            consecutive_l = 0
            for i in range(1, len(df) + 1):
                if pd.notna(psarl_series.iloc[-i]): consecutive_l += 1
                else: break
            if consecutive_l >= 3:
                sar_score = 1.0
                sar_signal = "MUA MẠNH (Bullish SAR 🟢)"
                sar_reason = f"Có {consecutive_l} chấm SAR nằm liên tiếp DƯỚI đường giá -> Xu hướng tăng mạnh."
            else:
                sar_score = 0.5
                sar_signal = "MUA CHỚM (Early Bullish 🟢)"
                sar_reason = f"Xuất hiện {consecutive_l} chấm SAR đầu tiên DƯỚI đường giá -> Tín hiệu đảo chiều sớm."
        elif pd.notna(psars_series.iloc[-1]):
            sar_val = psars_series.iloc[-1]
            consecutive_s = 0
            for i in range(1, len(df) + 1):
                if pd.notna(psars_series.iloc[-i]): consecutive_s += 1
                else: break
            if consecutive_s >= 3:
                sar_score = -1.0
                sar_signal = "BÁN MẠNH (Bearish SAR 🔴)"
                sar_reason = f"Có {consecutive_s} chấm SAR nằm liên tiếp TRÊN đường giá -> Xu hướng giảm mạnh."
            else:
                sar_score = -0.5
                sar_signal = "BÁN CHỚM (Early Bearish 🔴)"
                sar_reason = f"Xuất hiện {consecutive_s} chấm SAR đầu tiên TRÊN đường giá -> Rủi ro đảo chiều giảm sớm."

        # --- ICHIMOKU ---
        ichi_score, ichi_signal, ichi_reason = 0.0, "TRUNG TÍNH", ""
        kumo_top = max(isa_curr, isb_curr)
        kumo_bottom = min(isa_curr, isb_curr)

        if kumo_bottom <= close_curr <= kumo_top:
            ichi_signal = "THEO DÕI"
            ichi_reason = "Giá nằm luẩn quẩn bên trong mây Kumo (Chưa rõ xu hướng)."
            ichi_score = 0.0
        else:
            if close_curr > kumo_top:
                if tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr:
                    ichi_score = 2.0
                    ichi_signal = "MUA MẠNH (Ichimoku Bullish 🌸)"
                    ichi_reason = "Giá nằm TRÊN mây Kumo VÀ đường Tenkan cắt lên trên đường Kijun."
                else:
                    ichi_score = 1.0
                    ichi_signal = "MUA TÍCH CỰC (Above Kumo)"
                    ichi_reason = "Giá nằm TRÊN mây Kumo nhưng đường Tenkan và Kijun chưa có giao cắt mới."
            elif close_curr < kumo_bottom:
                if tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr:
                    ichi_score = -2.0
                    ichi_signal = "BÁN MẠNH (Ichimoku Bearish 🌧️)"
                    ichi_reason = "Giá nằm DƯỚI mây Kumo VÀ đường Tenkan cắt xuống dưới đường Kijun."
                else:
                    ichi_score = -1.0
                    ichi_signal = "TIÊU CỰC (Below Kumo)"
                    ichi_reason = "Giá nằm DƯỚI mây Kumo nhưng đường Tenkan và Kijun chưa có giao cắt mới."

        # --- VOLUME PROFILE ---
        vpvr_score = 0.0
        vpvr_signal = "THEO DÕI"
        vpvr_reason = f"Giá đi ngang ngay trong vùng biên của dải POC ({poc_bottom:,.1f} - {poc_top:,.1f})"
        if close_curr > poc_top:
            vpvr_score = 0.5
            vpvr_signal = "MUA (Buy 📈)"
            vpvr_reason = f"Giá đóng cửa ({close_curr:,}) vượt hẳn lên TRÊN vùng POC tích lũy khối lượng lớn ({poc_top:,.1f})"
        elif close_curr < poc_bottom:
            vpvr_score = -0.5
            vpvr_signal = "BÁN (Sell 📉)"
            vpvr_reason = f"Giá đóng cửa ({close_curr:,}) thủng xuống DƯỚI vùng POC ({poc_bottom:,.1f})"

        # --- VOLUME KICKER ---
        vol_kicker_score = 0.0
        vol_kicker_signal = "KHÔNG KÍCH HOẠT"
        vol_kicker_reason = "Khối lượng chưa chạm mốc bùng nổ dòng tiền."

        if vol_curr >= 1.5 * vol_ma20_curr:
            vol_kicker_score = 1.5
            vol_kicker_signal = "BÙNG NỔ DÒNG TIỀN MẠNH (Vol Kicker +1.5 🚀)"
            vol_kicker_reason = f"Vol phiên nay ({vol_curr:,.0f}) >= 1.5x Vol MA20 ({vol_ma20_curr:,.0f}) -> Dòng tiền lớn nhập cuộc cực mạnh!"
        elif vol_curr >= 1.2 * vol_ma20_curr:
            vol_kicker_score = 1.0
            vol_kicker_signal = "DÒNG TIỀN VÀO TÍCH CỰC (Vol Kicker +1.0 🟢)"
            vol_kicker_reason = f"Vol phiên nay ({vol_curr:,.0f}) >= 1.2x Vol MA20 ({vol_ma20_curr:,.0f}) -> Dòng tiền chớm gia tăng."

        base_score = round(bb_score + rsi_score + macd_score + sar_score + ichi_score + vpvr_score, 1)
        total_score = round(base_score + vol_kicker_score, 1)

        if total_score >= 6.0:
            final_action = "🚀 SIÊU SÓNG (Strong Buy)"
            alert_type = "success"
            final_meaning = "Đồng thuận 100% giữa Xu hướng & Dòng tiền | MUA CHỦ ĐỘNG / Mua gia tăng tối đa tỷ trọng"
        elif 3.5 <= total_score <= 5.5:
            final_action = "📈 TĂNG TRƯỞNG (Buy)"
            alert_type = "info"
            final_meaning = "Xu hướng tăng rõ nét, đa số chỉ báo ủng hộ | MUA THĂM DÒ / Mua khi có nhịp Rút chân (Pullback)"
        elif -1.0 <= total_score <= 3.0:
            final_action = "🎯 TÍCH LŨY / TRUNG TÍNH (Watch)"
            alert_type = "warning"
            final_meaning = "Giằng co Sideway hoặc Xung đột ngắn - trung hạn | RÌNH MUA (Đưa vào Watchlist chờ nổ Vol/BB)"
        elif -4.0 <= total_score <= -1.5:
            final_action = "⚠️ SUY YẾU (Caution)"
            alert_type = "error"
            final_meaning = "Cảnh báo vi phạm hỗ trợ ngắn hạn (MA20/Kijun) | HẠ TỶ TRỌNG / Ngừng mua mới hoàn toàn"
        else:
            final_action = "🔴 BÁN MẠNH (Strong Sell)"
            alert_type = "error"
            final_meaning = "Xu hướng giảm đồng loạt, bám biên dưới BB | BÁN DỨT KHÁT / Cắt lỗ, đứng ngoài bảo toàn vốn"

        return {
            "trade_date": df.index[-1].strftime('%Y-%m-%d'),
            "close_curr": close_curr,
            "base_score": base_score,
            "total_score": total_score,
            "final_action": final_action,
            "alert_type": alert_type,
            "final_meaning": final_meaning,
            "bb_signal": bb_signal, "bbu_curr": bbu_curr, "bbm_curr": bbm_curr, "bbl_curr": bbl_curr, 
            "bb_reason": bb_reason, "bb_score": bb_score, "pct_b_curr": pct_b_curr, "bw_curr": bw_curr,
            "bw_desc_status": bw_desc_status,
            "rsi_signal": rsi_signal, "rsi_curr": rsi_curr, "rsi_reason": rsi_reason, "rsi_score": rsi_score,
            "macd_signal": macd_signal, "macd_curr": macd_curr, "macds_curr": macds_curr, "macd_reasons": macd_reasons, "macd_score": macd_score,
            "sar_signal": sar_signal, "sar_val": sar_val, "sar_reason": sar_reason, "sar_score": sar_score,
            "ichi_signal": ichi_signal, "tenkan_curr": tenkan_curr, "kijun_curr": kijun_curr, "kumo_bottom": kumo_bottom, "kumo_top": kumo_top, "ichi_reason": ichi_reason, "ichi_score": ichi_score,
            "vpvr_signal": vpvr_signal, "poc_price": poc_price, "poc_bottom": poc_bottom, "poc_top": poc_top, "vpvr_reason": vpvr_reason, "vpvr_score": vpvr_score,
            "vol_kicker_signal": vol_kicker_signal, "vol_curr": vol_curr, "vol_ma20_curr": vol_ma20_curr, "vol_kicker_reason": vol_kicker_reason, "vol_kicker_score": vol_kicker_score
        }
    except Exception as e:
        return None

def render_detailed_report(res, symbol):
    """Hàm chuyên render giao diện đồ họa chi tiết từng Expander của một mã cổ phiếu cụ thể"""
    st.markdown("---")
    st.subheader(f"🎯 Kết quả phân tích: {symbol}")
    st.caption(f"Phiên giao dịch: **{res['trade_date']}** | Giá đóng cửa: **{res['close_curr']:,}**")
    
    if res['alert_type'] == "success": st.success(f"**{res['final_action']}** \n\n *Chiến thuật tác chiến:* {res['final_meaning']}")
    elif res['alert_type'] == "info": st.info(f"**{res['final_action']}** \n\n *Chiến thuật tác chiến:* {res['final_meaning']}")
    elif res['alert_type'] == "warning": st.warning(f"**{res['final_action']}** \n\n *Chiến thuật tác chiến:* {res['final_meaning']}")
    else: st.error(f"**{res['final_action']}** \n\n *Chiến thuật tác chiến:* {res['final_meaning']}")

    col1, col2 = st.columns(2)
    col1.metric("Điểm kỹ thuật gốc", f"{res['base_score']} đ")
    col2.metric("TỔNG ĐIỂM HỆ THỐNG", f"{res['total_score']} đ")

    st.markdown("### 📋 Chi tiết các chỉ báo kỹ thuật")

    # =========================================================
    # TRÌNH BÀY MỚI HIỂN THỊ TRẠNG THÁI NÉN / MỞ CHO BANDWIDTH
    # =========================================================
    with st.expander("📈 Bollinger Bands (Nâng cấp)"):
        st.write(f"- **Trạng thái:** `{res['bb_signal']}`")
        st.write(f"- **Khung giá dải BB:** Biên trên `{round(res['bbu_curr'],1)}` | MA20 `{round(res['bbm_curr'],1)}` | Biên dưới `{round(res['bbl_curr'],1)}`")
        
        st.markdown("**Đánh giá tương quan vị trí & biến động:**")
        st.write(f"  + **Mô tả kỹ thuật:** {res['bb_reason']}")
        st.write(f"  + **Vị trí tương quan (%B):** `{res['pct_b_curr']*100:.1f}%` *(>85%: Bám biên trên | <15%: Bám biên dưới | ~50%: Trục MA20)*")
        st.write(f"  + **Độ nở dải (Bandwidth):** `{round(res['bw_curr'], 3)}` ➡️ **{res['bw_desc_status']}**")
        st.write(f"- **Điểm số xếp hạng:** `{res['bb_score']} đ`")

    with st.expander("📉 Chỉ báo động lượng RSI(14)"):
        st.write(f"- **Trạng thái:** {res['rsi_signal']}")
        st.write(f"- **Chỉ số hiện tại:** `{round(res['rsi_curr'], 2)}`")
        st.write(f"- **Chi tiết:** {res['rsi_reason']}")
        st.write(f"- **Điểm số:** `{res['rsi_score']}`")

    with st.expander("📊 Chỉ báo xu hướng MACD"):
        st.write(f"- **Trạng thái:** {res['macd_signal']}")
        st.write(f"- **Giá trị:** MACD: `{round(res['macd_curr'], 3)}` | Signal: `{round(res['macds_curr'], 3)}`")
        for r in res['macd_reasons']:
            st.write(f"  + {r}")
        st.write(f"- **Điểm số:** `{res['macd_score']}`")

    with st.expander("⭐ Chỉ báo xu hướng Parabolic SAR"):
        st.write(f"- **Trạng thái:** {res['sar_signal']}")
        st.write(f"- **Giá trị SAR:** `{round(res['sar_val'], 1) if res['sar_val'] else 'N/A'}`")
        st.write(f"- **Chi tiết:** {res['sar_reason']}")
        st.write(f"- **Điểm số:** `{res['sar_score']}`")

    with st.expander("🌸 Hệ thống mây Ichimoku"):
        st.write(f"- **Trạng thái:** {res['ichi_signal']}")
        st.write(f"- **Cấu trúc:** Tenkan: `{round(res['tenkan_curr'], 1)}` | Kijun: `{round(res['kijun_curr'], 1)}` | Mây Kumo: `{round(res['kumo_bottom'], 1)} - {round(res['kumo_top'], 1)}`")
        st.write(f"- **Chi tiết:** {res['ichi_reason']}")
        st.write(f"- **Điểm số:** `{res['ichi_score']}`")

    with st.expander("📊 Khối lượng theo vùng giá VPVR (Volume Profile)"):
        st.write(f"- **Trạng thái:** {res['vpvr_signal']}")
        st.write(f"- **Đường giá POC:** `{res['poc_price']:,.1f}` | Biên (±0.5%): `{res['poc_bottom']:,.1f} - {res['poc_top']:,.1f}`")
        st.write(f"- **Chi tiết:** {res['vpvr_reason']}")
        st.write(f"- **Điểm số:** `{res['vpvr_score']}`")

    with st.expander("💡 Xác nhận dòng tiền - Volume Kicker"):
        st.write(f"- **Trạng thái:** {res['vol_kicker_signal']}")
        st.write(f"- **Khối lượng:** Phiên: `{res['vol_curr']:,.0f}` | Ngưỡng 1.2x - 1.5x Vol MA20: `{1.2 * res['vol_ma20_curr']:,.0f} - {1.5 * res['vol_ma20_curr']:,.0f}`")
        st.write(f"- **Chi tiết:** {res['vol_kicker_reason']}")
        st.write(f"- **Điểm bổ sung:** `{res['vol_kicker_score']}`")

# ========================================================
#                    GIAO DIỆN CHÍNH WEBAPP
# ========================================================
st.title("📊 Hệ Thống Phân Tích Cổ Phiếu Tự Động")

tab1, tab2 = st.tabs(["🔍 Phân Tích Đơn Lẻ", "📋 Rổ Cổ Phiếu Watchlist"])

# --------------------------------------------------------
# TAB 1: PHÂN TÍCH ĐƠN LẺ
# --------------------------------------------------------
with tab1:
    st.write("Nhập một mã chứng khoán Việt Nam bên dưới để bóc tách sức mạnh kỹ thuật đa tầng.")
    with st.form(key='ticker_form'):
        symbol = st.text_input("Nhập mã chứng khoán (Ví dụ: SSI, PVD, HPG):", value="SSI").strip().upper()
        submit_button = st.form_submit_button(label='Phân Tích Ngay')

    if submit_button and symbol:
        with st.spinner(f'Đang tải và tính toán dữ liệu cho mã {symbol}...'):
            df = fetch_data_mac(symbol, show_error=True)
            if df is not None:
                analysis_result = compute_signals(df)
                if analysis_result:
                    render_detailed_report(analysis_result, symbol)
                else:
                    st.error("[-] Không thể tính toán đầy đủ chỉ báo kỹ thuật cho mã này.")

# --------------------------------------------------------
# TAB 2: QUẢN LÝ & QUÉT RỔ 20 MÃ ĐỒNG LOẠT
# --------------------------------------------------------
with tab2:
    st.subheader("📋 Quản lý Rổ Watchlist (Tối đa 20 mã)")
    st.write("Thêm hoặc loại bỏ các mã trong danh mục của anh. Hệ thống sẽ quét toàn bộ và xếp hạng.")

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_ticker = st.text_input("Gõ mã CP mới muốn thêm vào rổ:", value="", key="add_ticker_input").strip().upper()
    with col_add2:
        st.write("##") 
        add_btn = st.button("➕ Thêm vào rổ", use_container_width=True)
        if add_btn and new_ticker:
            if new_ticker in st.session_state.watchlist:
                st.warning(f"Mã {new_ticker} đã tồn tại trong rổ của anh.")
            elif len(st.session_state.watchlist) >= 20:
                st.error("Rổ theo dõi đã đầy! Anh vui lòng xoá bớt mã trước khi thêm mới (Tối đa 20 mã).")
            else:
                st.session_state.watchlist.append(new_ticker)
                save_watchlist(st.session_state.watchlist)
                st.success(f"Đã thêm {new_ticker} thành công!")
                st.rerun()

    selected_watchlist = st.multiselect(
        "Danh sách rổ hiện tại (Bấm vào dấu 'X' để xoá nhanh mã khỏi rổ):",
        options=st.session_state.watchlist,
        default=st.session_state.watchlist
    )
    
    if selected_watchlist != st.session_state.watchlist:
        st.session_state.watchlist = selected_watchlist
        save_watchlist(st.session_state.watchlist)
        st.rerun()

    st.markdown("---")
        
    run_bulk = st.button("🚀 KÍCH HOẠT QUÉT ĐỒNG LOẠT BẢNG ĐIỂM RỔ DANH MỤC", use_container_width=True)
    
    if run_bulk:
        if not st.session_state.watchlist:
            st.warning("Rổ danh mục trống, anh vui lòng thêm ít nhất 1 mã để bắt đầu quét.")
        else:
            bulk_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            end_date = str(datetime.now().date())
            start_date = str((datetime.now() - timedelta(days=365)).date())
            
            def process_single_ticker(ticker):
                df_ticker = fetch_data_pure(ticker, start_date, end_date)
                if df_ticker is not None and not df_ticker.empty:
                    res_ticker = compute_signals(df_ticker)
                    if res_ticker:
                        return {
                            "Mã CP": ticker,
                            "Giá đóng cửa": f"{res_ticker['close_curr']:,}",
                            "Điểm Gốc": res_ticker['base_score'],
                            "TỔNG ĐIỂM": res_ticker['total_score'],
                            "Khuyến Nghị Tác Chiến": res_ticker['final_action']
                        }
                return {
                    "Mã CP": ticker, "Giá đóng cửa": "N/A", "Điểm Gốc": 0, "TỔNG ĐIỂM": -99, "Khuyến Nghị Tác Chiến": "⚠️ Lỗi kết nối / Dữ liệu trống"
                }

            total_tickers = len(st.session_state.watchlist)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(process_single_ticker, t): t for t in st.session_state.watchlist}
                
                for idx, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        bulk_results.append(result)
                    except Exception:
                        pass
                    
                    percent_complete = (idx + 1) / total_tickers
                    progress_bar.progress(percent_complete)
                    status_text.text(f"⚡ Đã quét xong: {ticker} ({idx+1}/{total_tickers})")
            
            progress_bar.empty()
            status_text.empty()
            
            if bulk_results:
                df_summary = pd.DataFrame(bulk_results)
                df_summary = df_summary.sort_values(by="TỔNG ĐIỂM", ascending=False).reset_index(drop=True)
                df_summary["TỔNG ĐIỂM"] = df_summary["TỔNG ĐIỂM"].apply(lambda x: "Lỗi" if x == -99 else f"{x} đ")
                df_summary["Điểm Gốc"] = df_summary["Điểm Gốc"].apply(lambda x: f"{x} đ")

                st.markdown("### 🏆 Bảng Xếp Hạng Khuyến Nghị Kỹ Thuật (Watchlist)")
                st.write("Bảng dữ liệu tự động lọc những mã có cấu trúc dòng tiền và kỹ thuật mạnh nhất lên đầu.")
                st.dataframe(df_summary, use_container_width=True)
            else:
                st.error("Không lấy được dữ liệu của bất kỳ mã nào trong danh mục.")
