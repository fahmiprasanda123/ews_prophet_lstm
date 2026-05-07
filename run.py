"""
Unified launcher for Agri-AI EWS v2.0.
Starts FastAPI server and Streamlit app together.
"""
import subprocess
import sys
import os
import signal
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def start_api(port=8000):
    """Start FastAPI server as a background process."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_DIR)


def start_streamlit(port=8501):
    """Start Streamlit app."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_DIR)


def main():
    print("=" * 60)
    print("🌾 Agri-AI EWS v2.0 — Starting Services")
    print("=" * 60)
    
    # Initialize database
    print("\n📦 Initializing database...")
    sys.path.insert(0, PROJECT_DIR)
    from data.database import get_store
    store = get_store()
    count = store.migrate_from_csv()
    print(f"   Database ready: {count} records")

    processes = []

    # Start FastAPI
    print("\n🚀 Starting API server on http://localhost:8000 ...")
    api_proc = start_api()
    processes.append(api_proc)

    time.sleep(2)

    # Start Streamlit
    print("🌐 Starting Streamlit on http://localhost:8501 ...")
    st_proc = start_streamlit()
    processes.append(st_proc)

    print("\n" + "=" * 60)
    print("✅ All services running!")
    print("   📊 Dashboard: http://localhost:8501")
    print("   🔌 API Docs:  http://localhost:8000/docs")
    print("   Press Ctrl+C to stop all services")
    print("=" * 60)

    def shutdown(signum, frame):
        print("\n\n🛑 Shutting down...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("   All services stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for processes
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
