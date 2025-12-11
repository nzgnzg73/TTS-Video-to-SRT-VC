#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║                     NOMI - System Manager                      ║
║              Advanced PC Control & Management Tool             ║
║                   No UAC - Full Auto Start                     ║
╚═══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import ctypes
import socket
import threading
import webbrowser
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# Install Dependencies
# ═══════════════════════════════════════════════════════════════

def install_package(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

required_packages = ['flask', 'flask-cors', 'psutil', 'pyngrok']

for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"📦 Installing {pkg}...")
        try:
            install_package(pkg)
        except:
            pass

import psutil
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# Try to import pyngrok for public URL
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
except:
    NGROK_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

PREFERRED_PORT = 21880
APP_NAME = "Nomi"
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
TEMPLATES_DIR = BASE_DIR / "templates"

TEMPLATES_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
CORS(app)

# ═══════════════════════════════════════════════════════════════
# Global Variables
# ═══════════════════════════════════════════════════════════════

PUBLIC_URL = None
CURRENT_PORT = PREFERRED_PORT

# ═══════════════════════════════════════════════════════════════
# Windows API
# ═══════════════════════════════════════════════════════════════

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

# ═══════════════════════════════════════════════════════════════
# Port Management
# ═══════════════════════════════════════════════════════════════

def is_port_available(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind(('0.0.0.0', port))
            return True
    except:
        return False

def kill_port_process(port):
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n'):
                if 'LISTENING' in line:
                    pid = int(line.strip().split()[-1])
                    if pid != os.getpid():
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, 
                                      capture_output=True)
                        print(f"🔪 Killed process {pid}")
                        return True
    except:
        pass
    return False

def find_available_port(start=21880):
    for port in range(start, start + 100):
        if is_port_available(port):
            return port
    return start

# ═══════════════════════════════════════════════════════════════
# State Manager
# ═══════════════════════════════════════════════════════════════

class StateManager:
    def __init__(self):
        self.config = self.load_config()
        self.screen_on_active = False
        self.screen_thread = None
        self.tracked_cmds = {}  # Store CMD outputs
        
    def load_config(self):
        default = {
            "firstRun": True,
            "screenOn": False,
            "autoStart": False,
            "theme": "light",
            "uacBypassed": False
        }
        
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    return {**default, **saved}
            except:
                pass
        return default
    
    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass

state = StateManager()

# ═══════════════════════════════════════════════════════════════
# Screen Control
# ═══════════════════════════════════════════════════════════════

def keep_screen_on_loop():
    while state.screen_on_active:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        except:
            pass
        time.sleep(30)

def start_screen_on():
    if not state.screen_on_active:
        state.screen_on_active = True
        state.config["screenOn"] = True
        state.save_config()
        threading.Thread(target=keep_screen_on_loop, daemon=True).start()
        print("🖥️ Screen Always On: ENABLED")

def stop_screen_on():
    state.screen_on_active = False
    state.config["screenOn"] = False
    state.save_config()
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except:
        pass
    print("🖥️ Screen Always On: DISABLED")

# ═══════════════════════════════════════════════════════════════
# Auto Start (Task Scheduler - NO UAC!)
# ═══════════════════════════════════════════════════════════════

def create_autostart_task():
    """Create Task Scheduler task - NO UAC POPUP!"""
    try:
        python_exe = sys.executable
        script_path = str(Path(__file__).absolute())
        working_dir = str(BASE_DIR)
        
        # XML for Task Scheduler
        xml_content = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Nomi System Manager - Auto Start</Description>
    <Author>Nomi</Author>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"{python_exe}"</Command>
      <Arguments>"{script_path}"</Arguments>
      <WorkingDirectory>{working_dir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''
        
        # Save XML file
        xml_path = BASE_DIR / "nomi_task.xml"
        with open(xml_path, 'w', encoding='utf-16') as f:
            f.write(xml_content)
        
        # Delete old task if exists
        subprocess.run(
            f'schtasks /delete /tn "NomiAutoStart" /f',
            shell=True, capture_output=True
        )
        
        # Create new task
        result = subprocess.run(
            f'schtasks /create /tn "NomiAutoStart" /xml "{xml_path}"',
            shell=True, capture_output=True, text=True
        )
        
        # Clean up XML
        try:
            xml_path.unlink()
        except:
            pass
        
        if result.returncode == 0:
            state.config["autoStart"] = True
            state.config["uacBypassed"] = True
            state.save_config()
            print("🚀 Auto-Start: ENABLED (No UAC)")
            return True
        else:
            print(f"❌ Task creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def remove_autostart_task():
    """Remove Task Scheduler task"""
    try:
        result = subprocess.run(
            'schtasks /delete /tn "NomiAutoStart" /f',
            shell=True, capture_output=True
        )
        state.config["autoStart"] = False
        state.save_config()
        print("🛑 Auto-Start: DISABLED")
        return True
    except:
        return False

def check_autostart_task():
    """Check if task exists"""
    try:
        result = subprocess.run(
            'schtasks /query /tn "NomiAutoStart"',
            shell=True, capture_output=True
        )
        return result.returncode == 0
    except:
        return False

# ═══════════════════════════════════════════════════════════════
# Public URL (ngrok)
# ═══════════════════════════════════════════════════════════════

def start_public_url(port):
    global PUBLIC_URL
    
    if not NGROK_AVAILABLE:
        print("⚠️ ngrok not available - Public URL disabled")
        return None
    
    try:
        # Kill any existing ngrok
        try:
            ngrok.kill()
        except:
            pass
        
        # Start tunnel
        tunnel = ngrok.connect(port, "http")
        PUBLIC_URL = tunnel.public_url
        print(f"🌐 Public URL: {PUBLIC_URL}")
        return PUBLIC_URL
    except Exception as e:
        print(f"⚠️ ngrok error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# System Info
# ═══════════════════════════════════════════════════════════════

def get_battery_info():
    try:
        battery = psutil.sensors_battery()
        if battery:
            time_left = ""
            if battery.secsleft > 0 and battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                hours = battery.secsleft // 3600
                mins = (battery.secsleft % 3600) // 60
                time_left = f"{hours}h {mins}m"
            
            return {
                "percent": int(battery.percent),
                "charging": battery.power_plugged and battery.percent < 100,
                "plugged": battery.power_plugged,
                "timeLeft": time_left
            }
    except:
        pass
    return {"percent": 100, "charging": False, "plugged": True, "timeLeft": "Desktop"}

def get_network_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def get_cmd_processes():
    """Get CMD/Console processes with details"""
    processes = []
    target = ['cmd.exe', 'powershell.exe', 'python.exe', 'pythonw.exe', 
              'node.exe', 'conhost.exe', 'windowsterminal.exe', 'pwsh.exe']
    
    my_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent']):
        try:
            info = proc.info
            if info['pid'] == my_pid:
                continue
            
            name = info.get('name', '').lower()
            if name in target:
                cmdline = info.get('cmdline', [])
                display_name = info['name']
                
                if cmdline and len(cmdline) > 1:
                    cmd_str = ' '.join(cmdline[1:])
                    if len(cmd_str) > 60:
                        cmd_str = cmd_str[:60] + "..."
                    display_name = f"{info['name']} → {cmd_str}"
                
                # Get port if possible
                port = None
                try:
                    conns = proc.connections()
                    for conn in conns:
                        if conn.status == 'LISTEN':
                            port = conn.laddr.port
                            break
                except:
                    pass
                
                processes.append({
                    "pid": info['pid'],
                    "name": display_name,
                    "port": port,
                    "cpu": round(info.get('cpu_percent', 0), 1)
                })
        except:
            pass
    
    return processes

# ═══════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET', 'POST'])
def api_state():
    if request.method == 'POST':
        data = request.json or {}
        if 'theme' in data:
            state.config['theme'] = data['theme']
        state.save_config()
        return jsonify({"success": True})
    
    return jsonify({
        "screenOn": state.screen_on_active,
        "autoStart": check_autostart_task(),
        "theme": state.config.get("theme", "light")
    })

@app.route('/api/first-run', methods=['GET', 'POST'])
def api_first_run():
    if request.method == 'POST':
        state.config["firstRun"] = False
        state.save_config()
        return jsonify({"success": True})
    return jsonify({"isFirstRun": state.config.get("firstRun", True)})

@app.route('/api/battery')
def api_battery():
    return jsonify(get_battery_info())

@app.route('/api/urls')
def api_urls():
    network_ip = get_network_ip()
    return jsonify({
        "local": f"http://localhost:{CURRENT_PORT}",
        "network": f"http://{network_ip}:{CURRENT_PORT}",
        "public": PUBLIC_URL
    })

@app.route('/api/screen-on', methods=['POST'])
def api_screen_on():
    data = request.json or {}
    if data.get('enabled'):
        start_screen_on()
    else:
        stop_screen_on()
    return jsonify({"success": True, "enabled": state.screen_on_active})

@app.route('/api/autostart', methods=['POST'])
def api_autostart():
    data = request.json or {}
    if data.get('enabled'):
        success = create_autostart_task()
    else:
        success = remove_autostart_task()
    return jsonify({"success": success, "enabled": check_autostart_task()})

@app.route('/api/power/<action>', methods=['POST'])
def api_power(action):
    def do_action():
        time.sleep(2)
        if action == 'shutdown':
            os.system('shutdown /s /t 0')
        elif action == 'restart':
            os.system('shutdown /r /t 0')
        elif action == 'restart_no_auto':
            remove_autostart_task()
            os.system('shutdown /r /t 0')
    
    if action in ['shutdown', 'restart', 'restart_no_auto']:
        threading.Thread(target=do_action, daemon=True).start()
        return jsonify({"success": True})
    
    return jsonify({"success": False})

@app.route('/api/cmd/list')
def api_cmd_list():
    return jsonify(get_cmd_processes())

@app.route('/api/cmd/kill/<int:pid>', methods=['POST'])
def api_cmd_kill(pid):
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        time.sleep(0.5)
        if proc.is_running():
            proc.kill()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/cmd/connect', methods=['POST'])
def api_cmd_connect():
    data = request.json or {}
    url = data.get('url', '')
    
    if not url:
        return jsonify({"success": False, "error": "No URL provided"})
    
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Nomi/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            return jsonify({
                "success": True,
                "status": resp.status,
                "output": content[:5000]
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/public-url/start', methods=['POST'])
def api_start_public():
    global PUBLIC_URL
    url = start_public_url(CURRENT_PORT)
    return jsonify({"success": url is not None, "url": url})

@app.route('/api/public-url/stop', methods=['POST'])
def api_stop_public():
    global PUBLIC_URL
    try:
        ngrok.kill()
        PUBLIC_URL = None
    except:
        pass
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def print_banner():
    global CURRENT_PORT
    network_ip = get_network_ip()
    battery = get_battery_info()
    auto = "✅ Enabled" if check_autostart_task() else "❌ Disabled"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     ███╗   ██╗ ██████╗ ███╗   ███╗██╗                           ║
║     ████╗  ██║██╔═══██╗████╗ ████║██║                           ║
║     ██╔██╗ ██║██║   ██║██╔████╔██║██║                           ║
║     ██║╚██╗██║██║   ██║██║╚██╔╝██║██║                           ║
║     ██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║                           ║
║     ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝                           ║
║                                                                  ║
║               ⚡ System Manager v2.0                             ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📍 Local:    http://localhost:{CURRENT_PORT}                         ║
║  🌐 Network:  http://{network_ip}:{CURRENT_PORT}                      ║
║  🔋 Battery:  {battery['percent']}% {'⚡ Charging' if battery['charging'] else ''}
║  🚀 Auto-Start: {auto}                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

def run():
    global CURRENT_PORT
    
    # Find available port
    if not is_port_available(PREFERRED_PORT):
        print(f"⚠️ Port {PREFERRED_PORT} busy, trying to free it...")
        kill_port_process(PREFERRED_PORT)
        time.sleep(1)
        
        if not is_port_available(PREFERRED_PORT):
            CURRENT_PORT = find_available_port(PREFERRED_PORT + 1)
            print(f"📌 Using port {CURRENT_PORT}")
        else:
            CURRENT_PORT = PREFERRED_PORT
    else:
        CURRENT_PORT = PREFERRED_PORT
    
    print_banner()
    
    # Restore screen state
    if state.config.get("screenOn"):
        start_screen_on()
    
    # Open browser
    def open_browser():
        time.sleep(2)
        webbrowser.open(f'http://localhost:{CURRENT_PORT}')
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start public URL in background
    if NGROK_AVAILABLE:
        threading.Thread(target=lambda: start_public_url(CURRENT_PORT), daemon=True).start()
    
    print(f"\n✅ Server running on port {CURRENT_PORT}")
    print("📌 Press Ctrl+C to stop\n")
    
    # Run Flask
    try:
        app.run(host='0.0.0.0', port=CURRENT_PORT, debug=False, threaded=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        stop_screen_on()