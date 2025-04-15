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
import logging

# Configura un logger di base (potresti volerlo più avanzato)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

COMFYUI_PATH = "/workspace/ComfyUI"

# Python executable path in the container
PYTHON_CMD = "python"
UV_CMD = "/workspace/ComfyUI/venv/bin/python"

# TCP server configuration
TCP_HOST = '0.0.0.0'  # Listen on all interfaces
TCP_PORT = 8080       # Port for TCP connection
MAX_CLIENTS = 20      # Maximum number of concurrent clients

def receive_files_in_background(one_time_code):
    """Task in background per ricevere file e processarli."""
    thread_name = threading.current_thread().name
    logging.info(f"[{thread_name}] Starting file receive task with code: {one_time_code}")
    receive_cmd = ["runpodctl", "receive", one_time_code]
    current_dir = os.getcwd() # Directory corrente dove runpodctl scarica i file
    
    try:
        logging.info(f"[{thread_name}] Executing command: {' '.join(receive_cmd)}")
        # Esegui il comando runpodctl
        # Potresti voler catturare stdout/stderr per un debug più dettagliato:
        # result = subprocess.run(receive_cmd, check=True, timeout=1800, capture_output=True, text=True)
        # logging.info(f"[{thread_name}] runpodctl stdout: {result.stdout}")
        # logging.error(f"[{thread_name}] runpodctl stderr: {result.stderr}") # Logga stderr anche in caso di successo se contiene warnings
        
        # Versione semplice:
        subprocess.run(receive_cmd, check=True, timeout=1800)
        logging.info(f"[{thread_name}] runpodctl receive command completed successfully for code: {one_time_code}")

        # --- Inizio Post-Processing ---
        logging.info(f"[{thread_name}] Starting post-processing in directory: {current_dir}")

        # Sposta model.safetensors se esiste
        source_model_path = os.path.join(current_dir, "model.safetensors")
        loras_dir = os.path.join(COMFYUI_PATH, "models", "loras")
        target_model_path = os.path.join(loras_dir, "model.safetensors") # O usa il nome originale se preferisci

        if os.path.exists(source_model_path):
            logging.info(f"[{thread_name}] Found '{source_model_path}'. Attempting to move to '{loras_dir}'")
            try:
                # Assicurati che la directory di destinazione esista
                os.makedirs(loras_dir, exist_ok=True)
                shutil.move(source_model_path, target_model_path)
                logging.info(f"[{thread_name}] Successfully moved '{source_model_path}' to '{target_model_path}'")
            except Exception as move_error:
                logging.error(f"[{thread_name}] Failed to move '{source_model_path}': {move_error}", exc_info=True)
        else:
            logging.warning(f"[{thread_name}] File '{source_model_path}' not found after runpodctl finished.")

        logging.info(f"[{thread_name}] Post-processing finished for code: {one_time_code}")
        # --- Fine Post-Processing ---

    except subprocess.CalledProcessError as e:
        logging.error(f"[{thread_name}] runpodctl command failed with return code {e.returncode} for code {one_time_code}.", exc_info=True)
        # Se hai catturato l'output, loggalo:
        # logging.error(f"[{thread_name}] runpodctl stdout: {e.stdout}")
        # logging.error(f"[{thread_name}] runpodctl stderr: {e.stderr}")
    except subprocess.TimeoutExpired:
        logging.error(f"[{thread_name}] runpodctl command timed out after 1800 seconds for code {one_time_code}.", exc_info=True)
    except Exception as e:
        # Cattura qualsiasi altra eccezione imprevista durante l'intero processo
        logging.error(f"[{thread_name}] An unexpected error occurred during receive/processing for code {one_time_code}: {e}", exc_info=True) # exc_info=True aggiunge il traceback

# Modifica receive_files_handler per usare la nuova funzione
def receive_files_handler(job):
    """Endpoint to handle file receiving via runpodctl"""
    job_input = job["input"]
    one_time_code = job_input.get("one_time_code")
    if not one_time_code:
        return {"status": "error", "message": "No one-time code provided"}

    # Avvia il thread in background passando il codice
    thread = threading.Thread(target=receive_files_in_background, args=(one_time_code,), name=f"Receive-{one_time_code}")
    thread.daemon = True # Permette al programma principale di uscire anche se questo thread è in esecuzione
    thread.start()
    
    return {
        "status": "success",
        "message": f"File transfer initiated with code: {one_time_code}. Processing continues in background. Check logs for details."
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