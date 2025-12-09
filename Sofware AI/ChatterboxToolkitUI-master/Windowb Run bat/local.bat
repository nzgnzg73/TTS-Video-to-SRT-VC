@echo off
REM --- یقینی بنائیں کہ یہ پاتھ (Path) آپ کے سسٹم پر درست ہے ---
call venv\scripts\activate
REM --- 'local.py' ہی چلائیں، کیونکہ ساری سیٹنگز اسی میں ہیں ---
python local.py
pause