import time
import psutil
import os
import sys
from datetime import datetime

def monitor_performance():
    """Monitor system performance during Flask app execution."""
    print("🔍 Performance Monitor Started")
    print("=" * 50)
    
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"📊 Performance Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 50)
            print(f"🖥️  CPU Usage: {cpu_percent}%")
            print(f"🧠 Memory: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
            print(f"💾 Disk: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
            print()
            
            if python_processes:
                print("🐍 Python Processes:")
                for proc in python_processes:
                    print(f"   PID {proc['pid']}: {proc['name']} - CPU: {proc['cpu_percent']:.1f}%, Memory: {proc['memory_percent']:.1f}%")
            else:
                print("🐍 No Python processes found")
            
            print()
            print("Press Ctrl+C to stop monitoring")
            
        except KeyboardInterrupt:
            print("\n👋 Performance monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_performance()
    except ImportError:
        print("❌ psutil not installed. Install with: pip install psutil")
        sys.exit(1)
