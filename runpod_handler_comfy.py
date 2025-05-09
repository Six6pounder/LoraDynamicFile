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
import requests
import hashlib

COMFYUI_PATH = "/workspace/ComfyUI"

# Python executable path in the container
PYTHON_CMD = "python"
UV_CMD = "/workspace/ComfyUI/venv/bin/python"

# TCP server configuration
TCP_HOST = '0.0.0.0'  # Listen on all interfaces
TCP_PORT = 8080       # Port for TCP connection
MAX_CLIENTS = 20      # Maximum number of concurrent clients

def monitor_and_move_lora_file():
    """Monitor for model.safetensors file and move it to loras directory when fully uploaded"""
    model_file_path = "model.safetensors"
    lora_dir = os.path.join(COMFYUI_PATH, "models/loras")
    
    # Ensure lora directory exists
    os.makedirs(lora_dir, exist_ok=True)
    
    print("Starting file monitor for model.safetensors...")
    
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        if os.path.exists(model_file_path):
            # File exists, check if it's being actively written
            print(f"Found model.safetensors")
            
            try:
                # Metodo più accurato: Calcolare un hash del file
                # Se l'hash rimane costante per alcuni controlli consecutivi, significa che
                # il file non sta più cambiando effettivamente.
                
                def get_file_hash(filepath):
                    """Calcola l'hash SHA-256 del file"""
                    try:
                        # Apriamo in modalità binaria e leggiamo a blocchi per file grandi
                        with open(filepath, 'rb') as f:
                            sha256_hash = hashlib.sha256()
                            for byte_block in iter(lambda: f.read(4096), b""):
                                sha256_hash.update(byte_block)
                            return sha256_hash.hexdigest()
                    except Exception as e:
                        print(f"Errore nel calcolo dell'hash: {e}")
                        return None
                
                # Controlla stabilità dell'hash
                stable_count = 0
                stable_threshold = 3  # Controllo 3 volte (intervalli di 10 secondi)
                last_hash = get_file_hash(model_file_path)
                file_size = os.path.getsize(model_file_path)
                print(f"File size: {file_size} bytes, Hash iniziale: {last_hash[:10]}...")
                
                while stable_count < stable_threshold:
                    time.sleep(10)  # Intervallo più lungo per file grandi
                    
                    # Verifica il file lock
                    try:
                        # Prova ad aprire il file in modalità esclusiva
                        # Se è ancora in uso da un altro processo, questo fallirà
                        with open(model_file_path, 'rb+') as lock_test:
                            current_hash = get_file_hash(model_file_path)
                            current_size = os.path.getsize(model_file_path)
                            
                            if current_hash == last_hash and current_size == file_size:
                                stable_count += 1
                                print(f"Contenuto stabile: controllo {stable_count}/{stable_threshold}")
                            else:
                                stable_count = 0
                                print(f"Contenuto cambiato, riavvio controllo")
                                last_hash = current_hash
                                file_size = current_size
                    except IOError:
                        # File ancora in uso da un altro processo
                        print("File ancora in uso da un altro processo, attendo...")
                        stable_count = 0
                
                # File è stabile, spostalo
                destination = os.path.join(lora_dir, os.path.basename(model_file_path))
                shutil.move(model_file_path, destination)
                print(f"File stabile per {stable_threshold * 10} secondi. Spostato in: {destination}")
            
            except Exception as e:
                print(f"Errore durante il monitoraggio o lo spostamento del file: {e}")
                # Attendi prima di riprovare in caso di errore
                time.sleep(30)

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
            # Run the receive command
            cmd = ["runpodctl", "receive", one_time_code]
            subprocess.run(cmd, check=True, timeout=1800)
            print(f"Files received successfully with code: {one_time_code}")
            
            # Non è più necessario controllare e spostare il file qui poiché lo fa
            # il monitor in background

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

def check_status_handler(job):
    # Check if ComfyUI is able to receive commands using /system_stats
    try:
        response = requests.get("http://localhost:8188/system_stats")
        if response.status_code == 200:
            return {
                "status": "success",
                "ready": True,
                "message": "ComfyUI is ready to receive commands"
            }
        else:
            return {
                "status": "error",
                "ready": False,
                "message": "ComfyUI is not ready to receive commands"
            }
    except Exception as e:
        return {
            "status": "error",
            "ready": False,
            "message": f"Error checking ComfyUI status: {str(e)}"
        }

def check_lora_handler(job):
    """Check if a specific LoRA model has been uploaded correctly"""
    job_input = job["input"]
    lora_name = job_input.get("lora_name")

    if not lora_name:
        return {
            "status": "error",
            "message": "No LoRA model name provided"
        }

    # Directory containing LoRA models
    lora_dir = "/workspace/ComfyUI/models/loras"
    
    if not os.path.exists(lora_dir):
        return {
            "status": "error",
            "message": "LoRA directory does not exist"
        }
    
    # Check if the specific LoRA file exists
    lora_path = os.path.join(lora_dir, lora_name)
    if os.path.exists(lora_path):
        # Get file size to verify it's not empty
        file_size = os.path.getsize(lora_path)
        
        return {
            "status": "success",
            "message": f"LoRA model '{lora_name}' found with size {file_size} bytes",
            "model_found": True,
            "file_size": file_size
        }
    else:
        # If the exact name wasn't found, check if there's a file that starts with the same name
        # (in case the file was renamed during upload)
        possible_matches = [f for f in os.listdir(lora_dir) if f.lower().startswith(lora_name.lower())]
        
        if possible_matches:
            match_details = []
            for match in possible_matches:
                match_path = os.path.join(lora_dir, match)
                match_size = os.path.getsize(match_path)
                match_details.append({"name": match, "size": match_size})
                
            return {
                "status": "success",
                "message": f"Found {len(possible_matches)} similar LoRA models",
                "model_found": True,
                "similar_models": match_details
            }
        else:
            # No matches found
            # List all available LoRA models
            available_models = os.listdir(lora_dir) if os.path.exists(lora_dir) else []
            
            return {
                "status": "error",
                "message": f"LoRA model '{lora_name}' not found",
                "model_found": False,
                "available_models": available_models[:20]  # Limit to first 20 to avoid huge responses
            }

# Map endpoints to handlers
handlers = {
    "receive_files": receive_files_handler,
    "shutdown": shutdown_handler,
    "check_upload": check_upload_handler,
    "status": check_status_handler,
    "check_lora": check_lora_handler
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

def start_comfyui():
    try:
        print("Starting ComfyUI...")
        # Log that we're about to start ComfyUI
        with open("/workspace/comfyui_startup.log", "w") as f:
            f.write("Starting ComfyUI process\n")
        subprocess.run([UV_CMD, "main.py", "--listen", "0.0.0.0"], cwd=COMFYUI_PATH, check=True)
    except Exception as e:
        print(f"Error starting ComfyUI: {e}")
        # Log the error
        with open("/workspace/comfyui_error.log", "w") as f:
            f.write(f"Error starting ComfyUI: {e}\n")

if __name__ == "__main__":
    # Create an event for signaling threads to stop
    stop_event = threading.Event()
    
    # Log that we're starting the handler
    print(f"RunPod Handler starting from {os.getcwd()}...")
    
    # Start the file monitor thread
    monitor_thread = threading.Thread(target=monitor_and_move_lora_file)
    monitor_thread.daemon = True
    monitor_thread.start()
    print("File monitor thread started")
    
    # Start TCP server in a separate thread
    tcp_thread = threading.Thread(target=start_tcp_server, args=(stop_event,))
    tcp_thread.daemon = True
    tcp_thread.start()
    print("TCP server thread started")
    
    # Start ComfyUI in a separate thread
    comfyui_thread = threading.Thread(target=start_comfyui)
    comfyui_thread.daemon = True
    comfyui_thread.start()
    print("ComfyUI thread started")
    
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