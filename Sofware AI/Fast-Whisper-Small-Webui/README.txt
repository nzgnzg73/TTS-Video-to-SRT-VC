<!-- NZG 73 Logo Footer -->
<div style="text-align: center; margin: 20px 0;">
    <pre style="background:#000; color:#0f0; padding:20px; font-family:monospace; display:inline-block; border-radius:10px;">
╔══════════════════════════════════════════════════════╗
║                                                      ║
║  ███╗   ██╗ ███████╗  ██████╗  ███████╗ ██████╗     ║
║  ████╗  ██║ ╚══███╔╝ ██╔════╝  ╚════██║ ╚════██╗    ║
║  ██╔██╗ ██║   ███╔╝  ██║  ███╗     ██╔╝  █████╔╝    ║
║  ██║╚██╗██║  ███╔╝   ██║   ██║    ██╔╝  ╚════██╗    ║
║  ██║ ╚████║ ███████╗ ╚██████╔╝    ██║   ██████╔╝    ║
║  ╚═╝  ╚═══╝ ╚══════╝  ╚═════╝     ╚═╝   ╚═════╝     ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    </pre>
</div>


        </ul>

    </div>
</div>


# 🎧 Fast Whisper WebUI (Standalone & Portable)
### AI Powered Audio to SRT/Text Transcriber with GPU Support

**Fast Whisper WebUI** is a lightweight, offline, and portable tool that converts Audio/Video files into Subtitles (SRT) and Text. It supports **100+ Languages** and allows you to switch between **GPU** and **CPU** effortlessly.

---

## 👤 Creator & Contact Information
**Developed by:** NZG 73  
🌐 **Website:** [nzg73.blogspot.com](https://nzg73.blogspot.com)  
📧 **Email:** nzgnzg73@gmail.com  
📺 **YouTube:** [NZG 73](https://youtube.com/@NZG73)

---

# 🇬🇧 How to Install (Step-by-Step Guide)

Follow these simple steps to install and run this tool on your Windows computer. This version includes an **Automatic Fix** for NVIDIA/DLL errors.

### ✅ Step 1: Install Python
First, download and install Python (Version 3.10 or 3.11 is recommended).
👉 **[Download Python Here](https://www.python.org/downloads/)**
> **⚠️ IMPORTANT:** During installation, make sure to check the box that says **"Add Python to PATH"**.

### ✅ Step 2: Download & Open Folder
1. Download this project (Clone or Zip).
2. Extract the folder.
3. Open the project folder where `app.py` is located.
4. Click on the address bar at the top, type `cmd`, and press **Enter**. (This opens the Command Prompt in the current folder).

### ✅ Step 3: Install Requirements (Magic Step)
Copy the command below and paste it into the Command Prompt (CMD). Press **Enter**.
This single command will install everything, including the **GPU Fix** and **NVIDIA Libraries**.

```bash
pip install -r requirements.txt

(Please wait for a few minutes while it downloads the necessary files).
✅ Step 4: Run the App
Once the installation is complete, type the following command to start the tool:
python app.py

✅ Step 5: Open in Browser
You will see a link in the CMD window. Open your browser (Chrome/Edge) and go to:
👉 http://localhost:2188
🇵🇰 انسٹال کرنے کا مکمل طریقہ (اردو میں)
یہ ٹول آپ کی آڈیو اور ویڈیو فائلز سے سب ٹائٹلز (SRT) بناتا ہے۔ یہ طریقہ نئے یوزرز کے لیے لکھا گیا ہے تاکہ آپ کو انسٹالیشن میں کوئی مسئلہ نہ آئے۔
✅ مرحلہ 1: پائیتھون انسٹال کریں (Python)
سب سے پہلے اپنے کمپیوٹر میں Python انسٹال کریں۔
👉 یہاں سے ڈاؤنلوڈ کریں
> ⚠️ ضروری نوٹ: انسٹال کرتے وقت ایک چھوٹا سا خانہ ہوگا "Add Python to PATH"، اس پر ٹک (✔) لازمی لگائیں ورنہ ٹول کام نہیں کرے گا۔
> 
✅ مرحلہ 2: فولڈر اور CMD اوپن کریں
 * اس پروجیکٹ کو ڈاؤنلوڈ کریں اور فولڈر اوپن کریں۔
 * جہاں app.py فائل پڑی ہے، اسی فولڈر میں اوپر ایڈریس بار (Address Bar) میں cmd لکھیں اور Enter کا بٹن دبائیں۔
 * آپ کے سامنے ایک کالی سکرین (Command Prompt) کھل جائے گی۔
✅ مرحلہ 3: ضروری فائلز انسٹال کریں (سب سے اہم)
نیچے دی گئی کمانڈ کو کاپی کریں اور کالی سکرین (CMD) میں پیسٹ کر کے Enter دبائیں۔
یہ کمانڈ خود بخود تمام سیٹنگز کر دے گی اور GPU کا مسئلہ بھی حل کر دے گی۔
pip install -r requirements.txt

(یہ انٹرنیٹ سے فائلز ڈاؤنلوڈ کرے گا، اس لیے تھوڑا انتظار کریں۔)
✅ مرحلہ 4: سافٹ ویئر چلائیں
جب انسٹالیشن مکمل ہو جائے، تو اسی کالی سکرین میں یہ لکھ کر Enter دبائیں:
python app.py

✅ مرحلہ 5: استعمال کریں
اب اپنے براؤزر (Chrome یا Edge) میں یہ لنک کھولیں:
👉 http://localhost:2188
🌟 Key Features (اہم خصوصیات)
 * 🚀 Run on GPU & CPU: Switch easily with radio buttons.
 * 📂 Portable Models: Models are saved inside the project folder (/Models), keeping your C: Drive clean.
 * 🌍 100+ Languages: Supports Urdu, English, Hindi, Arabic, Chinese, and many more.
 * 🛠 Multiple Models: Choose from Tiny, Base, Small, Medium, and Large-v3.
 * 💾 Auto Save: Automatically saves SRT and Text files in the Outputs folder.
Made with ❤️ by NZG 73