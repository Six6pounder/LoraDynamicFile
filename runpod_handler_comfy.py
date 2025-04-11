import runpod
import os
import json
import copy
import itertools
import subprocess
import shutil
from tqdm import tqdm
import time
import socket
import threading
import signal
import sys

COMFYUI_PATH = "/workspace/ComfyUI"

# Python executable path in the container
PYTHON_CMD = "python"
UV_CMD = "uv run"

# TCP server configuration
TCP_HOST = '0.0.0.0'  # Listen on all interfaces
TCP_PORT = 8080       # Port for TCP connection
MAX_CLIENTS = 5       # Maximum number of concurrent clients


# Path to the base configuration file
BASE_CONFIG_PATH = COMFYUI_PATH + "/wf.json"


def receive_files_handler(job):
    """Endpoint to handle file receiving via runpodctl"""
    job_input = job["input"]

    # Get the one-time code for receiving files
    one_time_code = job_input.get("one_time_code")
    if not one_time_code:
        return {
            "status": "error",
            "message": "No one-time code provided for file transfer"
        }
    
    # Start a background thread to receive the files
    def receive_files_in_background():
        try:
            # Change directory to loras model directory
            os.chdir(COMFYUI_PATH + "/models/loras")
            
            # Run the receive command
            cmd = ["runpodctl", "receive", one_time_code]
            subprocess.run(cmd, check=True, timeout=600)
            print(f"Files received successfully with code: {one_time_code}")
            
            # Find and extract the tar.gz file
            for file in os.listdir("."):
                if file.endswith(".tar.gz"):
                    print(f"Extracting archive: {file}")
                    subprocess.run(["tar", "-xzf", file], check=True)
                    
                    # Remove the tar file after extraction
                    os.remove(file)
                    print(f"Extracted contents and removed archive: {file}")
                    
                    # If a directory named 'temp_bundle' exists, move its contents up
                    if os.path.exists("temp_bundle"):
                        for item in os.listdir("temp_bundle"):
                            src_path = os.path.join("temp_bundle", item)
                            if os.path.exists(item):
                                if os.path.isdir(item):
                                    shutil.rmtree(item)
                                else:
                                    os.remove(item)
                            shutil.move(src_path, ".")
                        os.rmdir("temp_bundle")
                        print("Moved contents from temp_bundle to input directory")
                    break
        except Exception as e:
            print(f"Error receiving or extracting files: {e}")
    
    # Start background thread
    thread = threading.Thread(target=receive_files_in_background)
    thread.daemon = True
    thread.start()
    
    # Return immediately to avoid blocking
    return {
        "status": "success",
        "message": f"File transfer initiated with code: {one_time_code}. Transfer will continue in background. Old files will be deleted."
    }

def shutdown_handler(job):
    """Endpoint to shutdown the pod"""
    return {
        "status": "success",
        "message": "To shutdown the pod, use the RunPod API or web interface."
    }

def check_upload_handler(job):
    """Check the status of uploaded files"""
    input_dir = "/workspace/input"
    
    if not os.path.exists(input_dir):
        return {
            "status": "error",
            "message": "Input directory does not exist"
        }
    
    try:
        # Get all files in the input directory
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            relative_path = os.path.relpath(root, input_dir)
            if relative_path == ".":
                relative_path = ""
            
            for file in files:
                file_path = os.path.join(relative_path, file)
                all_files.append(file_path)
        
        # Count image files
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        return {
            "status": "success",
            "message": f"Found {len(all_files)} files in total, including {len(image_files)} image files",
            "files": all_files,
            "image_count": len(image_files)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking files: {str(e)}"
        }

# Map endpoints to handlers
handlers = {
    "receive_files": receive_files_handler,
    "shutdown": shutdown_handler,
    "check_upload": check_upload_handler
}

# RunPod serverless handler
def http_handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "")
    
    if action in handlers:
        return handlers[action](job)
    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. Available actions: {list(handlers.keys())}"
        }

# TCP server implementation
def handle_client(client_socket):
    """Handle client TCP connections"""
    try:
        # Receive data from client
        data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            data += chunk
            
            # Check if we have received a complete message
            if b"\n" in data:
                break
        
        # Decode and parse the request
        if not data:
            return
            
        try:
            job = json.loads(data.decode('utf-8'))
            action = job.get("input", {}).get("action", "")
            
            # Process the request
            if action in handlers:
                result = handlers[action](job)
            else:
                result = {
                    "status": "error",
                    "message": f"Unknown action: {action}. Available actions: {list(handlers.keys())}"
                }
                
            # Send the response back
            client_socket.sendall(json.dumps(result).encode('utf-8') + b"\n")
            
        except json.JSONDecodeError:
            # Send error response
            error = {
                "status": "error",
                "message": "Invalid JSON request"
            }
            client_socket.sendall(json.dumps(error).encode('utf-8') + b"\n")
            
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        # Close the connection
        client_socket.close()

def start_tcp_server(stop_event):
    """Start a TCP server to listen for client connections"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((TCP_HOST, TCP_PORT))
        server.listen(MAX_CLIENTS)
        print(f"TCP Server listening on {TCP_HOST}:{TCP_PORT}")
        
        # Set a timeout so we can check the stop event
        server.settimeout(1.0)
        
        while not stop_event.is_set():
            try:
                client, addr = server.accept()
                print(f"Accepted connection from {addr[0]}:{addr[1]}")
                
                # Handle client in a new thread
                client_thread = threading.Thread(target=handle_client, args=(client,))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if not stop_event.is_set():
                    print(f"Error accepting connection: {e}")
                continue
            
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.close()

if __name__ == "__main__":
    # Create an event for signaling threads to stop
    stop_event = threading.Event()
    
    # Start TCP server in a separate thread
    tcp_thread = threading.Thread(target=start_tcp_server, args=(stop_event,))
    tcp_thread.daemon = True
    tcp_thread.start()
    
    # Start ComfyUI in a separate thread
    def start_comfyui():
        try:
            print("Starting ComfyUI...")
            subprocess.run([UV_CMD, "main.py", "--listen", "0.0.0.0"], cwd=COMFYUI_PATH, check=True)
        except Exception as e:
            print(f"Error starting ComfyUI: {e}")
    
    comfyui_thread = threading.Thread(target=start_comfyui)
    comfyui_thread.daemon = True
    comfyui_thread.start()
    
    # Run the handler
    try:
        print("Starting server in production mode...")
        # Keep the server running
        while not stop_event.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down server...")
                break
    except Exception as e:
        print(f"Handler error: {e}")
    finally:
        stop_event.set() 