name: Daily Stock Scanner Bot

on:
  schedule:
    # 15:00 giờ Việt Nam (ICT) tương đương với 08:00 giờ Quốc Tế (UTC)
    # Chạy từ thứ 2 đến thứ 6 hàng tuần
    - cron: '0 8 * * 1-5'
  workflow_dispatch: # Cho phép anh ấn nút bắt chạy thủ công bất cứ lúc nào

jobs:
  run-scanner:
    runs-on: ubuntu-latest

    steps:
    # Đã nâng cấp từ v3 lên v4 để tương thích hoàn toàn với hệ thống mới
    - name: Checkout repository code
      uses: actions/checkout@v4

    # Đã nâng cấp từ v4 lên v5 để tương thích hoàn toàn với hệ thống mới
    - name: Set up Python environment
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: Install required libraries
      run: |
        python -m pip install --upgrade pip
        pip install pandas pandas-ta vnstock

    - name: Execute automated stock scan
      env:
        TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: python scan_alert.py
