#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║                     NOMI - System Manager                      ║
║              Advanced PC Control & Management Tool             ║
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
import winreg
from pathlib import Path

# Install dependencies if needed
def install_dependencies():
    required = ['flask', 'flask-cors', 'psutil']
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

install_dependencies()

import psutil
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

PREFERRED_PORT = 21880
APP_NAME = "Nomi"
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create templates directory if not exists
TEMPLATES_DIR.mkdir(exist_ok=True)

# Flask App
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
CORS(app)

# ═══════════════════════════════════════════════════════════════
# Port Management
# ═══════════════════════════════════════════════════════════════

def is_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind(('127.0.0.1', port))
            return True
    except (OSError, socket.error):
        return False

def find_available_port(start_port=21880, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None

def kill_port_process(port):
    """Kill process using the specified port"""
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    proc = psutil.Process(conn.pid)
                    proc.terminate()
                    print(f"🔪 Killed process {conn.pid} on port {port}")
                    return True
                except:
                    pass
    except:
        pass
    return False

# ═══════════════════════════════════════════════════════════════
# Windows API Constants
# ═══════════════════════════════════════════════════════════════

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

# ═══════════════════════════════════════════════════════════════
# State Management
# ═══════════════════════════════════════════════════════════════

class StateManager:
    def __init__(self):
        self.config = self.load_config()
        self.screen_on_active = False
        self.screen_thread = None
        self.current_port = PREFERRED_PORT
        
    def load_config(self):
        default = {
            "firstRun": True,
            "screenOn": False,
            "autoStart": False,
            "theme": "light"
        }
        
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return {**default, **json.load(f)}
            except:
                pass
        return default
    
    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save config: {e}")
    
    def get_state(self):
        return {
            "screenOn": self.screen_on_active,
            "autoStart": self.config.get("autoStart", False),
            "theme": self.config.get("theme", "light")
        }
    
    def set_state(self, data):
        if "theme" in data:
            self.config["theme"] = data["theme"]
        if "autoStart" in data:
            self.config["autoStart"] = data["autoStart"]
        self.save_config()

state_manager = StateManager()

# ═══════════════════════════════════════════════════════════════
# Screen Control Functions
# ═══════════════════════════════════════════════════════════════

def keep_screen_on():
    """Keep the screen on continuously"""
    while state_manager.screen_on_active:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        except:
            pass
        threading.Event().wait(30)

def start_screen_on():
    """Start keeping screen on"""
    if not state_manager.screen_on_active:
        state_manager.screen_on_active = True
        state_manager.config["screenOn"] = True
        state_manager.save_config()
        state_manager.screen_thread = threading.Thread(target=keep_screen_on, daemon=True)
        state_manager.screen_thread.start()
        print("🖥️ Screen lock: ENABLED")

def stop_screen_on():
    """Stop keeping screen on"""
    state_manager.screen_on_active = False
    state_manager.config["screenOn"] = False
    state_manager.save_config()
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except:
        pass
    print("🖥️ Screen lock: DISABLED")

# ═══════════════════════════════════════════════════════════════
# Auto Start Functions
# ═══════════════════════════════════════════════════════════════

def get_startup_command():
    """Get the command to run at startup"""
    python_exe = sys.executable
    script_path = str(Path(__file__).absolute())
    return f'"{python_exe}" "{script_path}"'

def enable_autostart():
    """Add to Windows startup registry"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, get_startup_command())
        winreg.CloseKey(key)
        state_manager.config["autoStart"] = True
        state_manager.save_config()
        print("🚀 Auto-start: ENABLED")
        return True
    except Exception as e:
        print(f"❌ Error enabling autostart: {e}")
        return False

def disable_autostart():
    """Remove from Windows startup registry"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, APP_NAME)
        winreg.CloseKey(key)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Warning: {e}")
    
    state_manager.config["autoStart"] = False
    state_manager.save_config()
    print("🛑 Auto-start: DISABLED")
    return True

def check_autostart():
    """Check if autostart is enabled"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_READ
        )
        winreg.QueryValueEx(key, APP_NAME)
        winreg.CloseKey(key)
        return True
    except:
        return False

# ═══════════════════════════════════════════════════════════════
# System Functions
# ═══════════════════════════════════════════════════════════════

def get_battery_info():
    """Get battery information"""
    try:
        battery = psutil.sensors_battery()
        if battery:
            time_left = ""
            if battery.secsleft > 0 and battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                hours = battery.secsleft // 3600
                mins = (battery.secsleft % 3600) // 60
                time_left = f"{hours}h {mins}m left"
            
            return {
                "percent": int(battery.percent),
                "charging": battery.power_plugged and battery.percent < 100,
                "plugged": battery.power_plugged,
                "timeLeft": time_left
            }
    except:
        pass
    return {"percent": 100, "charging": False, "plugged": True, "timeLeft": "Desktop PC"}

def get_network_ip():
    """Get local network IP"""
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
    """Get all CMD/Console processes"""
    cmd_processes = []
    target_names = ['cmd.exe', 'powershell.exe', 'python.exe', 'pythonw.exe', 
                    'node.exe', 'conhost.exe', 'windowsterminal.exe']
    
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pinfo = proc.info
            if pinfo['pid'] == current_pid:
                continue
                
            if pinfo['name'] and pinfo['name'].lower() in target_names:
                name = pinfo['name']
                cmdline = pinfo.get('cmdline', [])
                if cmdline and len(cmdline) > 1:
                    cmd_str = ' '.join(cmdline[1:])[:50]
                    name = f"{name} - {cmd_str}"
                cmd_processes.append({
                    "pid": pinfo['pid'],
                    "name": name
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return cmd_processes

# ═══════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET', 'POST'])
def api_state():
    if request.method == 'POST':
        data = request.json
        state_manager.set_state(data)
        return jsonify({"success": True})
    return jsonify(state_manager.get_state())

@app.route('/api/first-run', methods=['GET', 'POST'])
def api_first_run():
    if request.method == 'POST':
        state_manager.config["firstRun"] = False
        state_manager.save_config()
        return jsonify({"success": True})
    return jsonify({"isFirstRun": state_manager.config.get("firstRun", True)})

@app.route('/api/battery')
def api_battery():
    return jsonify(get_battery_info())

@app.route('/api/urls')
def api_urls():
    network_ip = get_network_ip()
    port = state_manager.current_port
    return jsonify({
        "local": f"http://localhost:{port}",
        "network": f"http://{network_ip}:{port}",
        "public": None
    })

@app.route('/api/screen-on', methods=['POST'])
def api_screen_on():
    data = request.json
    enabled = data.get('enabled', False)
    
    if enabled:
        start_screen_on()
    else:
        stop_screen_on()
    
    return jsonify({"success": True, "enabled": state_manager.screen_on_active})

@app.route('/api/autostart', methods=['POST'])
def api_autostart():
    data = request.json
    enabled = data.get('enabled', False)
    
    if enabled:
        success = enable_autostart()
    else:
        success = disable_autostart()
    
    return jsonify({"success": success, "enabled": check_autostart()})

@app.route('/api/power/<action>', methods=['POST'])
def api_power(action):
    if action == 'shutdown':
        threading.Thread(target=lambda: os.system('shutdown /s /t 3'), daemon=True).start()
        return jsonify({"success": True, "message": "Shutting down in 3 seconds..."})
    
    elif action == 'restart':
        threading.Thread(target=lambda: os.system('shutdown /r /t 3'), daemon=True).start()
        return jsonify({"success": True, "message": "Restarting in 3 seconds..."})
    
    elif action == 'restart_no_auto':
        disable_autostart()
        threading.Thread(target=lambda: os.system('shutdown /r /t 3'), daemon=True).start()
        return jsonify({"success": True, "message": "Restarting without auto-start..."})
    
    return jsonify({"success": False, "message": "Unknown action"})

@app.route('/api/cmd/list')
def api_cmd_list():
    return jsonify(get_cmd_processes())

@app.route('/api/cmd/kill/<int:pid>', methods=['POST'])
def api_cmd_kill(pid):
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/cmd/connect', methods=['POST'])
def api_cmd_connect():
    data = request.json
    url = data.get('url', '')
    
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as response:
            content = response.read().decode('utf-8', errors='ignore')[:2000]
            return jsonify({
                "success": True,
                "output": f"Status: {response.status}\n\n{content}"
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

def print_banner(port, network_ip):
    """Print startup banner"""
    battery = get_battery_info()
    autostart = "Enabled" if check_autostart() else "Disabled"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ███╗   ██╗ ██████╗ ███╗   ███╗██╗                        ║
║     ████╗  ██║██╔═══██╗████╗ ████║██║                        ║
║     ██╔██╗ ██║██║   ██║██╔████╔██║██║                        ║
║     ██║╚██╗██║██║   ██║██║╚██╔╝██║██║                        ║
║     ██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║                        ║
║     ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝                        ║
║                                                               ║
║            ⚡ System Manager - Running on Port {port}          ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║  📍 Local URL:    http://localhost:{port}                      ║
║  🌐 Network URL:  http://{network_ip}:{port}              ║
╠═══════════════════════════════════════════════════════════════╣
║  🔋 Battery: {battery['percent']}%  |  🚀 Auto-Start: {autostart}                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def open_browser(port):
    """Open browser after short delay"""
    import time
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{port}')

def run_server():
    """Main function to run the server"""
    port = PREFERRED_PORT
    network_ip = get_network_ip()
    
    # Check if preferred port is available
    if not is_port_available(port):
        print(f"⚠️ Port {port} is not available!")
        
        # Try to kill the process using the port
        print(f"🔄 Trying to free port {port}...")
        if kill_port_process(port):
            import time
            time.sleep(1)
        
        # Check again
        if not is_port_available(port):
            # Find alternative port
            alt_port = find_available_port(port + 1)
            if alt_port:
                print(f"✅ Using alternative port: {alt_port}")
                port = alt_port
            else:
                print("❌ No available ports found!")
                input("Press Enter to exit...")
                return
    
    state_manager.current_port = port
    
    print_banner(port, network_ip)
    
    # Restore screen on state if was enabled
    if state_manager.config.get("screenOn", False):
        start_screen_on()
    
    # Open browser in background
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    print(f"✅ Server starting on port {port}...")
    print("📌 Press Ctrl+C to stop\n")
    
    # Run Flask
    try:
        from werkzeug.serving import run_simple
        run_simple(
            '0.0.0.0',
            port,
            app,
            use_reloader=False,
            use_debugger=False,
            threaded=True
        )
    except Exception as e:
        print(f"\n❌ Server Error: {e}")
        print("\n🔧 Possible solutions:")
        print("   1. Run as Administrator")
        print("   2. Check if antivirus is blocking")
        print("   3. Try a different port")
        input("\nPress Enter to exit...")

if __name__ == '__main__':
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n\n👋 Nomi stopped. Goodbye!")
        stop_screen_on()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to exit...")
