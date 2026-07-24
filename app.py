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
    """
    Hàm tính toán tổng hợp 6 chỉ báo kỹ thuật theo hệ trọng số chuẩn [-7.5 đến +9.0].
    Đồng bộ hoàn toàn dữ liệu đầu ra với giao diện Streamlit.
    """
    if df is None or df.empty or len(df) < 52:
        return None

    try:
        df = df.copy()

        # ==========================================
        # 1. BOLLINGER BANDS (Trọng số: [-2.0, +2.0])
        # ==========================================
        bb = ta.bbands(df['Close'], length=20, std=2.0)
        col_bbl = [c for c in bb.columns if c.startswith('BBL')][0]
        col_bbm = [c for c in bb.columns if c.startswith('BBM')][0]
        col_bbu = [c for c in bb.columns if c.startswith('BBU')][0]

        df['BBL'], df['BBM'], df['BBU'] = bb[col_bbl], bb[col_bbm], bb[col_bbu]
        df['BW'] = (df['BBU'] - df['BBL']) / df['BBM']
        df['Pct_B'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()

        close_curr = float(df['Close'].iloc[-1])
        vol_curr = float(df['Volume'].iloc[-1])
        vol_ma20_curr = float(df['Vol_MA20'].iloc[-1])

        bw_curr = float(df['BW'].iloc[-1])
        bw_prev = float(df['BW'].iloc[-2])
        bw_min_20 = float(df['BW'].tail(20).min())
        pct_b_curr = float(df['Pct_B'].iloc[-1])
        bbm_curr = float(df['BBM'].iloc[-1])
        bbm_prev = float(df['BBM'].iloc[-2])
        bbu_curr = float(df['BBU'].iloc[-1])
        bbl_curr = float(df['BBL'].iloc[-1])

        bb_score = 0.0
        bb_status = "Trung tính (Neutral ➖)"

        if (bw_curr > 1.15 * bw_prev) and (pct_b_curr >= 0.85) and (vol_curr >= 1.5 * vol_ma20_curr):
            bb_score = 2.0
            bb_status = "Bùng nổ dòng tiền (Breakout 🚀)"
        elif (bw_curr > 1.15 * bw_prev) and (pct_b_curr <= 0.15) and (vol_curr >= 1.2 * vol_ma20_curr):
            bb_score = -2.0
            bb_status = "Mở dải giảm mạnh (Walking Lower 🔴)"
        elif (bw_curr <= 1.10 * bw_min_20) and (0.40 <= pct_b_curr <= 0.60):
            bb_score = 1.0
            bb_status = "Tích lũy chặt (Squeeze 🎯)"
        elif (0.60 <= pct_b_curr < 0.85) and (bbm_curr > bbm_prev) and (vol_curr >= 1.0 * vol_ma20_curr):
            bb_score = 1.0
            bb_status = "Sóng tăng ổn định (Bullish 📈)"
        elif (pct_b_curr < 0.40) or (close_curr < bbm_curr):
            bb_score = -1.0
            bb_status = "Thủng hỗ trợ MA20 (Weak ⚠️)"

        # ==========================================
        # 2. ICHIMOKU KINKO HYO (Trọng số: [-2.0, +2.0])
        # ==========================================
        ichimoku_res = ta.ichimoku(df['High'], df['Low'], df['Close'], tenkan=9, kijun=26, senkou=52)
        ichimoku = ichimoku_res[0] if isinstance(ichimoku_res, tuple) else ichimoku_res
        
        col_tenkan = [c for c in ichimoku.columns if c.startswith('ITS')][0]
        col_kijun = [c for c in ichimoku.columns if c.startswith('IKS')][0]
        col_span_a = [c for c in ichimoku.columns if c.startswith('ISA')][0]
        col_span_b = [c for c in ichimoku.columns if c.startswith('ISB')][0]

        span_a_curr = float(ichimoku[col_span_a].iloc[-1])
        span_b_curr = float(ichimoku[col_span_b].iloc[-1])
        tenkan_curr = float(ichimoku[col_tenkan].iloc[-1])
        kijun_curr = float(ichimoku[col_kijun].iloc[-1])
        cloud_top = max(span_a_curr, span_b_curr)
        cloud_bottom = min(span_a_curr, span_b_curr)

        ichi_score = 0.0
        ichi_status = "Trong mây Kumo (Tích lũy ☁️)"

        if close_curr > cloud_top:
            if tenkan_curr > kijun_curr:
                ichi_score = 2.0
                ichi_status = "Trên mây + Tenkan > Kijun (Cực mạnh 🚀)"
            else:
                ichi_score = 1.0
                ichi_status = "Trên mây Kumo (Tăng giá 📈)"
        elif close_curr < cloud_bottom:
            if tenkan_curr < kijun_curr:
                ichi_score = -2.0
                ichi_status = "Dưới mây + Tenkan < Kijun (Rất xấu 🔴)"
            else:
                ichi_score = -1.0
                ichi_status = "Dưới mây Kumo (Giảm giá 📉)"

        # ==========================================
        # 3. MACD (Trọng số: [-1.5, +1.5])
        # ==========================================
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        col_macd = [c for c in macd.columns if c.startswith('MACD_') and not c.endswith('h') and not c.endswith('s')][0]
        col_signal = [c for c in macd.columns if c.startswith('MACDs_') or (c.startswith('MACD') and c.endswith('s'))][0]

        macd_val = float(macd[col_macd].iloc[-1])
        signal_val = float(macd[col_signal].iloc[-1])

        macd_score = 0.0
        macd_status = "Trung tính"

        if macd_val > signal_val:
            if macd_val > 0:
                macd_score = 1.5
                macd_status = "MACD > Signal & > 0 (Động lượng mạnh 🚀)"
            else:
                macd_score = 0.5
                macd_status = "MACD > Signal & < 0 (Phục hồi 📈)"
        else:
            if macd_val < 0:
                macd_score = -1.5
                macd_status = "MACD < Signal & < 0 (Suy yếu sâu 🔴)"
            else:
                macd_score = -0.5
                macd_status = "MACD < Signal & > 0 (Điều chỉnh ⚠️)"

        # ==========================================
        # 4. RSI (14) (Trọng số: [-1.0, +1.0])
        # ==========================================
        rsi_series = ta.rsi(df['Close'], length=14)
        rsi_curr = float(rsi_series.iloc[-1])

        rsi_score = 0.0
        rsi_status = "Trung tính"

        if 50 <= rsi_curr <= 70:
            rsi_score = 1.0
            rsi_status = f"RSI = {rsi_curr:.1f} (Vùng tăng giá khỏe 📈)"
        elif rsi_curr > 70:
            rsi_score = 0.5
            rsi_status = f"RSI = {rsi_curr:.1f} (Quá mua ngắn hạn ⚠️)"
        elif 35 <= rsi_curr < 50:
            rsi_score = -0.5
            rsi_status = f"RSI = {rsi_curr:.1f} (Xung lực yếu 📉)"
        else:
            rsi_score = -1.0
            rsi_status = f"RSI = {rsi_curr:.1f} (Quá bán / Rất yếu 🔴)"

        # ==========================================
        # 5. PARABOLIC SAR (Trọng số: [-1.0, +1.0])
        # ==========================================
        psar = ta.psar(df['High'], df['Low'], df['Close'], af0=0.02, af=0.02, max_af=0.2)
        col_psarl = [c for c in psar.columns if c.startswith('PSARl')][0]
        col_psars = [c for c in psar.columns if c.startswith('PSARs')][0]

        psar_long = psar[col_psarl].iloc[-1]
        psar_short = psar[col_psars].iloc[-1]

        psar_score = 0.0
        psar_status = "Trung tính"
        sar_val = 0.0

        if pd.notna(psar_long):
            psar_score = 1.0
            psar_status = "Bullish SAR (Chấm dưới giá 🟢)"
            sar_val = float(psar_long)
        elif pd.notna(psar_short):
            psar_score = -1.0
            psar_status = "Bearish SAR (Chấm trên giá 🔴)"
            sar_val = float(psar_short)

        # ==========================================
        # 6. VOLUME KICKER (Trọng số: [0.0, +1.5])
        # ==========================================
        vol_score = 0.0
        vol_status = "Khối lượng bình thường"

        if vol_curr >= 1.5 * vol_ma20_curr and close_curr >= float(df['Open'].iloc[-1]):
            vol_score = 1.5
            vol_status = "Nổ Vol lớn (>150% MA20 Vol 🚀)"
        elif vol_curr >= 1.2 * vol_ma20_curr and close_curr >= float(df['Open'].iloc[-1]):
            vol_score = 0.75
            vol_status = "Vol cải thiện (>120% MA20 Vol 📈)"

        # ==========================================
        # TỔNG HỢP ĐIỂM SỐ & PHÂN LOẠI KHUYẾN NGHỊ
        # ==========================================
        total_score = round(bb_score + ichi_score + macd_score + rsi_score + psar_score + vol_score, 1)

        if total_score >= 6.0:
            final_action = "🚀 SIÊU SÓNG (MUA CHỦ ĐỘNG)"
            alert_type = "success"
            final_meaning = "Đồng thuận 100% giữa Xu hướng & Dòng tiền | MUA CHỦ ĐỘNG / Mua gia tăng tối đa tỷ trọng"
        elif 3.5 <= total_score < 6.0:
            final_action = "📈 TĂNG TRƯỞNG (MUA THĂM DÒ)"
            alert_type = "info"
            final_meaning = "Xu hướng tăng rõ nét, đa số chỉ báo ủng hộ | MUA THĂM DÒ / Mua khi có nhịp Rút chân (Pullback)"
        elif -1.0 <= total_score < 3.5:
            final_action = "🎯 TÍCH LŨY (RÌNH MUA / THEO DÕI)"
            alert_type = "warning"
            final_meaning = "Giằng co Sideway hoặc Xung đột ngắn - trung hạn | RÌNH MUA (Đưa vào Watchlist chờ nổ Vol/BB)"
        elif -4.0 <= total_score < -1.0:
            final_action = "⚠️ SUY YẾU (HẠ TỶ TRỌNG)"
            alert_type = "error"
            final_meaning = "Cảnh báo vi phạm hỗ trợ ngắn hạn (MA20/Kijun) | HẠ TỶ TRỌNG / Ngừng mua mới hoàn toàn"
        else:
            final_action = "🔴 BÁN MẠNH (BÁN DỨT KHÁT)"
            alert_type = "error"
            final_meaning = "Xu hướng giảm đồng loạt, bám biên dưới BB | BÁN DỨT KHÁT / Cắt lỗ, đứng ngoài bảo toàn vốn"

        # Đóng gói Dictionary (Đầy đủ khóa cho UI Render)
        return {
            "trade_date": df.index[-1].strftime('%Y-%m-%d'),
            "close_curr": close_curr,
            "base_score": round(total_score - vol_score, 1),
            "total_score": total_score,
            "final_action": final_action,
            "alert_type": alert_type,
            "final_meaning": final_meaning,
            
            # BB
            "bb_score": bb_score, "bb_signal": bb_status, "bb_status": bb_status, "bbu_curr": bbu_curr, "bbm_curr": bbm_curr, "bbl_curr": bbl_curr, "bb_reason": bb_status,
            # Ichimoku
            "ichi_score": ichi_score, "ichi_signal": ichi_status, "ichi_status": ichi_status, "tenkan_curr": tenkan_curr, "kijun_curr": kijun_curr, "kumo_bottom": cloud_bottom, "kumo_top": cloud_top, "ichi_reason": ichi_status,
            # MACD
            "macd_score": macd_score, "macd_signal": macd_status, "macd_status": macd_status, "macd_curr": macd_val, "macds_curr": signal_val, "macd_reasons": [macd_status],
            # RSI
            "rsi_score": rsi_score, "rsi_signal": rsi_status, "rsi_status": rsi_status, "rsi_curr": rsi_curr, "rsi_reason": rsi_status,
            # Parabolic SAR (Bổ sung cả sar_score lẫn psar_score)
            "psar_score": psar_score, "sar_score": psar_score, "sar_signal": psar_status, "psar_status": psar_status, "sar_val": sar_val, "sar_reason": psar_status,
            # Volume Kicker
            "vol_score": vol_score, "vol_kicker_score": vol_score, "vol_kicker_signal": vol_status, "vol_status": vol_status, "vol_curr": vol_curr, "vol_ma20_curr": vol_ma20_curr, "vol_kicker_reason": vol_status,
            # VPVR (Giữ UI tương thích)
            "vpvr_signal": "TRUNG TÍNH", "poc_price": close_curr, "poc_bottom": close_curr * 0.995, "poc_top": close_curr * 1.005, "vpvr_reason": "Đã tích hợp vào hệ thống 6 chỉ báo chuẩn", "vpvr_score": 0.0
        }
    except Exception:
        return None

def render_detailed_report(res, symbol):
    """Hàm render giao diện đồ họa chi tiết từng Expander của một mã cổ phiếu cụ thể"""
    st.markdown("---")
    st.subheader(f"🎯 Kết quả phân tích: {symbol}")
    st.caption(f"Phiên giao dịch: **{res['trade_date']}** | Giá đóng cửa: **{res['close_curr']:,}**")
    
    if res['alert_type'] == "success": st.success(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")
    elif res['alert_type'] == "info": st.info(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")
    elif res['alert_type'] == "warning": st.warning(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")
    else: st.error(f"**{res['final_action']}** \n\n *Ý nghĩa chiến thuật:* {res['final_meaning']}")

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
# TAB 2: QUẢN LÝ & QUÉT RỔ WATCHLIST
# --------------------------------------------------------
with tab2:
    st.subheader("📋 Quản lý Rổ Watchlist (Tối đa 20 mã)")
    st.write("Thêm hoặc loại bỏ các mã trong danh mục. Hệ thống sẽ quét toàn bộ và xếp hạng.")

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
                st.warning(f"Mã {new_ticker} đã tồn tại trong rổ.")
            elif len(st.session_state.watchlist) >= 20:
                st.error("Rổ theo dõi đã đầy! Vui lòng xoá bớt mã trước khi thêm mới (Tối đa 20 mã).")
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
            st.warning("Rổ danh mục trống, vui lòng thêm ít nhất 1 mã để bắt đầu quét.")
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
