import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
from vnstock.ui import Market
from datetime import datetime, timedelta
import concurrent.futures

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

def compute_signals(df):
    """Hàm lõi chuyên tính toán toán học và trả về bộ chỉ báo + điểm số kỹ thuật (Không render UI)"""
    try:
        # 1. Tính toán các chỉ báo bằng pandas-ta-classic
        bbands = df.ta.bbands(length=20, std=2)
        rsi = df.ta.rsi(length=14)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        psar = df.ta.psar(af0=0.02, af=0.02, max_af=0.2)
        ichi_df = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52) # Đã sửa không unpack
        
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

        # --- KHẢO SÁT CHỈ BÁO 1: BOLLINGER BANDS ---
        bb_score = 0
        bb_signal = "THEO DÕI ĐỨNG NGOÀI"
        bb_reason = "Giá đang biến động ổn định bên trong dải Bollinger"
        if close_prev <= bbm_prev and close_curr > bbm_curr:
            bb_score = 1
            bb_signal = "MUA (Buy)"
            bb_reason = "Giá đóng cửa cắt lên đường giữa MA20 (Vào trend tăng ngắn hạn)"
        elif low_curr <= bbl_curr and close_curr > bbl_curr:
            bb_score = 1
            bb_signal = "MUA (Buy)"
            bb_reason = "Giá chạm biên dưới (Lower Band) và rút chân bật lên thành công"
        elif close_prev >= bbm_prev and close_curr < bbm_curr:
            bb_score = -1
            bb_signal = "BÁN (Sell)"
            bb_reason = "Giá đóng cửa cắt xuống đường giữa MA20 (Gãy trend tăng ngắn hạn)"
        elif high_curr >= bbu_curr and close_curr < bbu_curr:
            bb_score = -1
            bb_signal = "BÁN (Sell)"
            bb_reason = "Giá chạm biên trên (Upper Band) và gặp áp lực quay đầu giảm"

        # --- KHẢO SÁT CHỈ BÁO 2: RSI ---
        rsi_score = 0
        rsi_signal = "TRUNG TÍNH"
        rsi_reason = "RSI đang biến động ở vùng an toàn, chưa vi phạm quá mua/quá bán"
        if rsi_curr <= 30:
            rsi_score = 1
            rsi_signal = "QUÁ BÁN (Oversold)"
            rsi_reason = f"RSI lọt vào vùng QUÁ BÁN ({round(rsi_curr, 1)} <= 30) -> Áp lực bán suy kiệt"
        elif rsi_prev < 50 and rsi_curr >= 50:
            rsi_score = 1
            rsi_signal = "TÍCH CỰC (Bullish)"
            rsi_reason = "RSI cắt lên mốc trung vị 50 -> Phe Mua đang lấy lại thế trận"
        elif rsi_curr >= 70:
            rsi_score = -1
            rsi_signal = "QUÁ MUA (Overbought)"
            rsi_reason = f"RSI lọt vào vùng QUÁ MUA ({round(rsi_curr, 1)} >= 70) -> Lực mua quá tải"
        elif rsi_prev > 50 and rsi_curr <= 50:
            rsi_score = -1
            rsi_signal = "TIÊU CỰC (Bearish)"
            rsi_reason = "RSI cắt xuống mốc trung vị 50 -> Phe Bán đang kiểm soát thế trận"

        # --- KHẢO SÁT CHỈ BÁO 3: MACD ---
        macd_score = 0
        macd_signal = "TRUNG TÍNH"
        macd_reasons = []
        if macd_prev <= macds_prev and macd_curr > macds_curr:
            macd_score += 1
            macd_signal = "MUA (Giao cắt Vàng)"
            macd_reasons.append("Đường MACD cắt lên trên đường Tín hiệu (Signal Line)")
        elif macd_prev >= macds_prev and macd_curr < macds_curr:
            macd_score -= 1
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
                macd_score += 1
                macd_signal = "MUA MẠNH (Phân kỳ dương 📈)"
                macd_reasons.append(f"PHÂN KỲ DƯƠNG: Giá phá đáy cũ ({df.index[last_trough_idx].strftime('%Y-%m-%d')}) nhưng MACD tạo đáy sau cao hơn.")
        if last_peak_idx is not None:
            if close_curr > safe_val('Close', last_peak_idx) and macd_curr < safe_val(macd_col, last_peak_idx):
                macd_score -= 1
                macd_signal = "BÁN MẠNH (Phân kỳ âm 📉)"
                macd_reasons.append(f"PHÂN KỲ ÂM: Giá vượt đỉnh cũ ({df.index[last_peak_idx].strftime('%Y-%m-%d')}) nhưng MACD tạo đỉnh sau thấp hơn.")
        if not macd_reasons:
            macd_reasons.append("Đường MACD và Tín hiệu đi ngang ổn định.")

        # --- KHẢO SÁT CHỈ BÁO 4: PARABOLIC SAR ---
        sar_score, sar_signal, sar_reason, sar_val = 0, "TRUNG TÍNH", "Không xác định rõ trạng thái.", 0
        psarl_series = df[psarl_col].iloc[:, 0] if isinstance(df[psarl_col], pd.DataFrame) else df[psarl_col]
        psars_series = df[psars_col].iloc[:, 0] if isinstance(df[psars_col], pd.DataFrame) else df[psars_col]

        if pd.notna(psarl_series.iloc[-1]):
            sar_val = psarl_series.iloc[-1]
            consecutive_l = 0
            for i in range(1, len(df) + 1):
                if pd.notna(psarl_series.iloc[-i]): consecutive_l += 1
                else: break
            if consecutive_l >= 3:
                sar_score = 2
                sar_signal = "MUA MẠNH (Bullish SAR 🟢)"
                sar_reason = f"Có {consecutive_l} chấm SAR nằm liên tiếp DƯỚI đường giá -> Xu hướng tăng mạnh."
            else:
                sar_score = 1
                sar_signal = "MUA CHỚM (Early Bullish 🟢)"
                sar_reason = f"Xuất hiện {consecutive_l} chấm SAR đầu tiên DƯỚI đường giá -> Tín hiệu đảo chiều sớm."
        elif pd.notna(psars_series.iloc[-1]):
            sar_val = psars_series.iloc[-1]
            consecutive_s = 0
            for i in range(1, len(df) + 1):
                if pd.notna(psars_series.iloc[-i]): consecutive_s += 1
                else: break
            if consecutive_s >= 3:
                sar_score = -2
                sar_signal = "BÁN MẠNH (Bearish SAR 🔴)"
                sar_reason = f"Có {consecutive_s} chấm SAR nằm liên tiếp TRÊN đường giá -> Xu hướng giảm mạnh."
            else:
                sar_score = -1
                sar_signal = "BÁN CHỚM (Early Bearish 🔴)"
                sar_reason = f"Xuất hiện {consecutive_s} chấm SAR đầu tiên TRÊN đường giá -> Rủi ro đảo chiều giảm sớm."

        # --- KHẢO SÁT CHỈ BÁO 5: ICHIMOKU ---
        ichi_score, ichi_signal, ichi_reason = 0, "TRUNG TÍNH", ""
        kumo_top = max(isa_curr, isb_curr)
        kumo_bottom = min(isa_curr, isb_curr)

        if kumo_bottom <= close_curr <= kumo_top:
            ichi_signal = "THEO DÕI"
            ichi_reason = "Giá nằm luẩn quẩn bên trong mây Kumo (Chưa rõ xu hướng)."
        else:
            if close_curr > kumo_top and (tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr):
                ichi_score = 1
                ichi_signal = "MUA (Ichimoku Bullish 🌸)"
                ichi_reason = "Giá nằm TRÊN mây Kumo VÀ đường Tenkan cắt lên trên đường Kijun."
            elif close_curr < kumo_bottom and (tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr):
                ichi_score = -1
                ichi_signal = "BÁN (Ichimoku Bearish 🌧️)"
                ichi_reason = "Giá nằm DƯỚI mây Kumo VÀ đường Tenkan cắt xuống dưới đường Kijun."
            else:
                ichi_reason = "Giá nằm TRÊN mây Kumo" if close_curr > kumo_top else "Giá nằm DƯỚI mây Kumo"
                ichi_reason += " nhưng đường Tenkan và Kijun chưa có giao cắt mới."

        # --- KHẢO SÁT CHỈ BÁO 6: VOLUME PROFILE (VPVR) ---
        vpvr_score = 0
        vpvr_signal = "THEO DÕI"
        vpvr_reason = f"Giá đi ngang ngay trong vùng biên của dải POC ({poc_bottom:,.1f} - {poc_top:,.1f})"
        if close_curr > poc_top:
            vpvr_score = 1
            vpvr_signal = "MUA (Buy 📈)"
            vpvr_reason = f"Giá đóng cửa ({close_curr:,}) vượt hẳn lên TRÊN vùng POC tích lũy khối lượng lớn ({poc_top:,.1f})"
        elif close_curr < poc_bottom:
            vpvr_score = -1
            vpvr_signal = "BÁN (Sell 📉)"
            vpvr_reason = f"Giá đóng cửa ({close_curr:,}) thủng xuống DƯỚI vùng POC ({poc_bottom:,.1f})"

        # --- VOLUME KICKER ---
        vol_ma20_series = volume_series.rolling(window=20).mean()
        vol_curr = safe_val('Volume', -1)
        vol_ma20_curr = float(vol_ma20_series.iloc[-1])
        
        base_score = bb_score + rsi_score + macd_score + sar_score + ichi_score + vpvr_score
        vol_kicker_score = 0
        vol_kicker_signal = "KHÔNG KÍCH HOẠT"
        vol_kicker_reason = "Khối lượng bình thường hoặc hệ thống chưa có ưu thế rõ ràng."

        if vol_curr > 1.3 * vol_ma20_curr:
            if base_score > 0:
                vol_kicker_score = 1
                vol_kicker_signal = "XÁC NHẬN BÙNG NỔ DÒNG TIỀN (Vol Kicker +1 🚀)"
                vol_kicker_reason = f"Vol phiên nay ({vol_curr:,.0f}) > 1.3x Vol MA20 ({vol_ma20_curr:,.0f}) kết hợp ƯU THẾ MUA -> Dòng tiền đẩy giá quyết liệt!"
            elif base_score < 0:
                vol_kicker_score = -1
                vol_kicker_signal = "XÁC NHẬN ÁP LỰC THÁO CHẠY (Vol Kicker -1 💥)"
                vol_kicker_reason = f"Vol phiên nay ({vol_curr:,.0f}) > 1.3x Vol MA20 ({vol_ma20_curr:,.0f}) kết hợp ƯU THẾ BÁN -> Dòng tiền lớn tháo hàng gấp!"

        total_score = base_score + vol_kicker_score
        
        if 6 <= total_score <= 9:
            final_action, alert_type = "🟩 MUA MẠNH (Strong Buy)", "success"
            final_meaning = "Đồng thuận xu hướng tăng tuyệt đối. Các chỉ báo lớn vào phom, động lượng mạnh, có dòng tiền lớn xác nhận."
        elif 2 <= total_score <= 5:
            final_action, alert_type = "🟨 MUA THĂM DÒ / GIỮ (Buy/Hold)", "info"
            final_meaning = "Xu hướng dịch chuyển sang tăng nhưng có thể đang thiếu volume hoặc gặp cản nhẹ. Phù hợp mua rải hoặc nắm giữ."
        elif -1 <= total_score <= 1:
            final_action, alert_type = "⬜ ĐỨNG NGOÀI THEO DÕI (Neutral)", "warning"
            final_meaning = "Các chỉ báo triệt tiêu lẫn nhau hoặc cổ phiếu đang đi ngang tích lũy không rõ xu hướng."
        elif -5 <= total_score <= -2:
            final_action, alert_type = "🟨 HẠ TỶ TRỌNG / NGỪNG MUA (Weak Sell)", "error"
            final_meaning = "Xu hướng ngắn hạn bắt đầu suy yếu, chớm thủng các mốc hỗ trợ. Tuyệt đối không gia tăng vị thế mua mới."
        else:
            final_action, alert_type = "🟥 BÁN QUYẾT LIỆT (Strong Sell)", "error"
            final_meaning = "Kích hoạt trạng thái quản trị rủi ro tối đa. Gãy xu hướng đồng loạt trên nhiều chỉ báo mạnh đi kèm vol lớn."

        # Đóng gói toàn bộ kết quả vào một Dictionary
        return {
            "trade_date": df.index[-1].strftime('%Y-%m-%d'),
            "close_curr": close_curr,
            "base_score": base_score,
            "total_score": total_score,
            "final_action": final_action,
            "alert_type": alert_type,
            "final_meaning": final_meaning,
            "bb_signal": bb_signal, "bbu_curr": bbu_curr, "bbm_curr": bbm_curr, "bbl_curr": bbl_curr, "bb_reason": bb_reason, "bb_score": bb_score,
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
    
    # Hiển thị Khuyến nghị Tổng hợp nổi bật
    if res['alert_type'] == "success": st.success(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")
    elif res['alert_type'] == "info": st.info(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")
    elif res['alert_type'] == "warning": st.warning(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")
    else: st.error(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")

    # Cột hiển thị điểm nhanh
    col1, col2 = st.columns(2)
    col1.metric("Điểm kỹ thuật gốc", f"{res['base_score']} đ")
    col2.metric("TỔNG ĐIỂM (Có Vol Kicker)", f"{res['total_score']} đ")

    st.markdown("### 📋 Chi tiết các chỉ báo kỹ thuật")

    with st.expander("📈 Bollinger Bands"):
        st.write(f"- **Trạng thái:** {res['bb_signal']}")
        st.write(f"- **Mốc BB:** Biên Trên: `{round(res['bbu_curr'],1)}` | Đường Giữa: `{round(res['bbm_curr'],1)}` | Biên Dưới: `{round(res['bbl_curr'],1)}`")
        st.write(f"- **Chi tiết:** {res['bb_reason']}")
        st.write(f"- **Điểm số:** `{res['bb_score']}`")

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
        st.write(f"- **Khối lượng:** Phiên: `{res['vol_curr']:,.0f}` | Ngưỡng 1.3x Vol MA20: `{1.3 * res['vol_ma20_curr']:,.0f}`")
        st.write(f"- **Chi tiết:** {res['vol_kicker_reason']}")
        st.write(f"- **Điểm bổ sung:** `{res['vol_kicker_score']}`")


# ========================================================
#                    GIAO DIỆN CHÍNH WEBAPP
# ========================================================
st.title("📊 Hệ Thống Phân Tích Cổ Phiếu Tự Động")

# Phân chia không gian làm việc bằng Tabs thành 2 phần độc lập
tab1, tab2 = st.tabs(["🔍 Phân Tích Đơn Lẻ", "📋 Rổ Cổ Phiếu Watchlist"])

# --------------------------------------------------------
# TAB 1: PHÂN TÍCH ĐƠN LẺ (GIỮ NGUYÊN TOÀN BỘ CODE CŨ)
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
# TAB 2: TÍNH NĂNG MỞ RỘNG - QUẢN LÝ & QUÉT RỔ 20 MÃ ĐỒNG LOẠT
# --------------------------------------------------------
with tab2:
    st.subheader("📋 Quản lý Rổ Watchlist (Tối đa 20 mã)")
    st.write("Thêm hoặc loại bỏ các mã trong danh mục của anh. Hệ thống sẽ quét toàn bộ và xếp hạng.")

    # Khởi tạo bộ nhớ danh mục mặc định ban đầu nếu chạy lần đầu tiên
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["SSI", "HPG", "VND", "PVD", "FPT", "DIG", "MWG"]

    # Khung thêm mã mới vào rổ
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_ticker = st.text_input("Gõ mã CP mới muốn thêm vào rổ:", value="", key="add_ticker_input").strip().upper()
    with col_add2:
        st.write("##") # Đồng bộ khoảng cách dòng với Input
        add_btn = st.button("➕ Thêm vào rổ", use_container_width=True)
        if add_btn and new_ticker:
            if new_ticker in st.session_state.watchlist:
                st.warning(f"Mã {new_ticker} đã tồn tại trong rổ của anh.")
            elif len(st.session_state.watchlist) >= 20:
                st.error("Rổ theo dõi đã đầy! Anh vui lòng xoá bớt mã trước khi thêm mới (Tối đa 20 mã).")
            else:
                st.session_state.watchlist.append(new_ticker)
                st.success(f"Đã thêm {new_ticker} thành công!")
                st.rerun()

    # Khung hiển thị trực quan và xóa nhanh bằng st.multiselect
    selected_watchlist = st.multiselect(
        "Danh sách rổ hiện tại (Bấm vào dấu 'X' để xoá nhanh mã khỏi rổ):",
        options=st.session_state.watchlist,
        default=st.session_state.watchlist
    )
    
    # Đồng bộ lại bộ nhớ nếu người dùng click xóa bớt mã bằng dấu X
    if selected_watchlist != st.session_state.watchlist:
        st.session_state.watchlist = selected_watchlist
        st.rerun()

    st.markdown("---")
    
    # Nút bấm quét tổng lực rổ cổ phiếu
    run_bulk = st.button("🚀 KÍCH HOẠT QUÉT ĐỒNG LOẠT BẢNG ĐIỂM RỔ DANH MỤC", use_container_width=True)
    
    if run_bulk:
        if not st.session_state.watchlist:
            st.warning("Rổ danh mục trống, anh vui lòng thêm ít nhất 1 mã để bắt đầu quét.")
        else:
            bulk_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Đóng gói tác vụ của 1 mã CP vào một hàm nhỏ để giao cho luồng (thread) xử lý
            def process_single_ticker(ticker):
                df_ticker = fetch_data_mac(ticker, show_error=False)
                if df_ticker is not None:
                    res_ticker = compute_signals(df_ticker)
                    if res_ticker:
                        return {
                            "Mã CP": ticker,
                            "Giá đóng cửa": f"{res_ticker['close_curr']:,}",
                            "Điểm Gốc": res_ticker['base_score'],
                            "TỔNG ĐIỂM": res_ticker['total_score'],
                            "Khuyến Nghị Tác Chiến": res_ticker['final_action']
                        }
                # Nếu lỗi mạng hoặc lỗi tính toán
                return {
                    "Mã CP": ticker, "Giá đóng cửa": "N/A", "Điểm Gốc": 0, "TỔNG ĐIỂM": -99, "Khuyến Nghị Tác Chiến": "⚠️ Lỗi kết nối / dữ liệu"
                }

            # KÍCH HOẠT ĐA LUỒNG (Mở 5 luồng chạy song song cùng lúc)
            total_tickers = len(st.session_state.watchlist)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Gửi toàn bộ mã CP cho 5 luồng xử lý
                future_to_ticker = {executor.submit(process_single_ticker, t): t for t in st.session_state.watchlist}
                
                # Gom kết quả ngay khi có bất kỳ luồng nào chạy xong 1 mã
                for idx, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        bulk_results.append(result)
                    except Exception as exc:
                        pass
                    
                    # Cập nhật thanh tiến trình mượt mà
                    percent_complete = (idx + 1) / total_tickers
                    progress_bar.progress(percent_complete)
                    status_text.text(f"⚡ Đã quét xong: {ticker} ({idx+1}/{total_tickers})")
            
            progress_bar.empty()
            status_text.empty()
            
            # --- RENDER BẢNG ĐIỂM ĐỒNG LOẠT ---
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
