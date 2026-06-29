import streamlit as st
import pandas as pd
import pandas_ta_classic as ta
from vnstock.ui import Market
from datetime import datetime, timedelta

# Thiết lập cấu hình trang hiển thị chuẩn Mobile-First
st.set_page_config(
    page_title="Hệ Thống Phân Tích Kỹ Thuật Cổ Phiếu",
    page_icon="📊",
    layout="centered"
)

def fetch_data_mac(symbol):
    """Kéo dữ liệu 1 năm bằng Unified UI"""
    end_date = str(datetime.now().date())
    start_date = str((datetime.now() - timedelta(days=365)).date())
    
    try:
        mkt = Market()
        df = mkt.equity(symbol).ohlcv(start=start_date, end=end_date)
        
        if df is None or df.empty:
            st.error(f"[-] Không tìm thấy dữ liệu hoặc mã {symbol} không tồn tại.")
            return None
        
        df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"[-] Lỗi kết nối hệ thống với mã {symbol}: {e}")
        return None

def advanced_analyzer(df, symbol):
    """Hệ thống phân tích độc lập và chấm điểm tổng hợp hiển thị lên giao diện Web"""
    # 1. Tính toán các chỉ báo bằng pandas-ta
    bbands = df.ta.bbands(length=20, std=2)
    rsi = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    psar = df.ta.psar(af0=0.02, af=0.02, max_af=0.2)
    ichi_df = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52)
    
    if bbands is None or rsi is None or macd is None or psar is None or ichi_df is None:
        st.error("[-] Không thể tính toán đầy đủ các chỉ báo kỹ thuật.")
        return
        
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

    # ==========================================
    # GIAO DIỆN HIỂN THỊ TRÊN WEB (STREAMLIT)
    # ==========================================
    st.markdown("---")
    st.subheader(f"🎯 Kết quả phân tích: {symbol}")
    st.caption(f"Phiên giao dịch: **{df.index[-1].strftime('%Y-%m-%d')}** | Giá đóng cửa: **{close_curr:,}**")
    
    # Hiển thị Khuyến nghị Tổng hợp nổi bật
    if alert_type == "success": st.success(f"**{final_action}** \n\n *Ý nghĩa chiến thuật:* {final_meaning}")
    elif alert_type == "info": st.info(f"**{final_action}** \n\n *Ý nghĩa chiến thuật:* {final_meaning}")
    elif alert_type == "warning": st.warning(f"**{final_action}** \n\n *Ý nghĩa chiến thuật:* {final_meaning}")
    else: st.error(f"**{final_action}** \n\n *Ý nghĩa chiến thuật:* {final_meaning}")

    # Cột hiển thị điểm nhanh
    col1, col2 = st.columns(2)
    col1.metric("Điểm kỹ thuật gốc", f"{base_score} đ")
    col2.metric("TỔNG ĐIỂM (Có Vol Kicker)", f"{total_score} đ")

    st.markdown("### 📋 Chi tiết các chỉ báo kỹ thuật")

    # Sử dụng Expander để cuộn trang trên điện thoại không bị quá dài
    with st.expander("📈 Bollinger Bands"):
        st.write(f"- **Trạng thái:** {bb_signal}")
        st.write(f"- **Mốc BB:** Biên Trên: `{round(bbu_curr,1)}` | Đường Giữa: `{round(bbm_curr,1)}` | Biên Dưới: `{round(bbl_curr,1)}`")
        st.write(f"- **Chi tiết:** {bb_reason}")
        st.write(f"- **Điểm số:** `{bb_score}`")

    with st.expander("📉 Chỉ báo động lượng RSI(14)"):
        st.write(f"- **Trạng thái:** {rsi_signal}")
        st.write(f"- **Chỉ số hiện tại:** `{round(rsi_curr, 2)}`")
        st.write(f"- **Chi tiết:** {rsi_reason}")
        st.write(f"- **Điểm số:** `{rsi_score}`")

    with st.expander("📊 Chỉ báo xu hướng MACD"):
        st.write(f"- **Trạng thái:** {macd_signal}")
        st.write(f"- **Giá trị:** MACD: `{round(macd_curr, 3)}` | Signal: `{round(macds_curr, 3)}`")
        for r in macd_reasons:
            st.write(f"  + {r}")
        st.write(f"- **Điểm số:** `{macd_score}`")

    with st.expander("⭐ Chỉ báo xu hướng Parabolic SAR"):
        st.write(f"- **Trạng thái:** {sar_signal}")
        st.write(f"- **Giá trị SAR:** `{round(sar_val, 1) if sar_val else 'N/A'}`")
        st.write(f"- **Chi tiết:** {sar_reason}")
        st.write(f"- **Điểm số:** `{sar_score}`")

    with st.expander("🌸 Hệ thống mây Ichimoku"):
        st.write(f"- **Trạng thái:** {ichi_signal}")
        st.write(f"- **Cấu trúc:** Tenkan: `{round(tenkan_curr, 1)}` | Kijun: `{round(kijun_curr, 1)}` | Mây Kumo: `{round(kumo_bottom, 1)} - {round(kumo_top, 1)}`")
        st.write(f"- **Chi tiết:** {ichi_reason}")
        st.write(f"- **Điểm số:** `{ichi_score}`")

    with st.expander("📊 Khối lượng theo vùng giá VPVR (Volume Profile)"):
        st.write(f"- **Trạng thái:** {vpvr_signal}")
        st.write(f"- **Đường giá POC:** `{poc_price:,.1f}` | Biên (±0.5%): `{poc_bottom:,.1f} - {poc_top:,.1f}`")
        st.write(f"- **Chi tiết:** {vpvr_reason}")
        st.write(f"- **Điểm số:** `{vpvr_score}`")

    with st.expander("💡 Xác nhận dòng tiền - Volume Kicker"):
        st.write(f"- **Trạng thái:** {vol_kicker_signal}")
        st.write(f"- **Khối lượng:** Phiên: `{vol_curr:,.0f}` | Ngưỡng 1.3x Vol MA20: `{1.3 * vol_ma20_curr:,.0f}`")
        st.write(f"- **Chi tiết:** {vol_kicker_reason}")
        st.write(f"- **Điểm bổ sung:** `{vol_kicker_score}`")

# --- HÀM MAIN KHỞI CHẠY WEBAPP ---
st.title("📊 Hệ Thống Phân Tích Cổ Phiếu Tự Động")
st.write("Nhập mã chứng khoán Việt Nam bên dưới để kiểm tra sức mạnh kỹ thuật đa tầng.")

# Form nhập mã cổ phiếu gọn gàng
with st.form(key='ticker_form'):
    symbol = st.text_input("Nhập mã chứng khoán (Ví dụ: SSI, PVD, HPG):", value="SSI").strip().upper()
    submit_button = st.form_submit_button(label='Phân Tích Ngay')

if submit_button and symbol:
    with st.spinner(f'Đang tải và tính toán dữ liệu cho mã {symbol}...'):
        df = fetch_data_mac(symbol)
        if df is not None:
            advanced_analyzer(df, symbol)
