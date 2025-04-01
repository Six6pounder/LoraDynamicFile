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

# Base path for OneTrainer in the container
ONETRAINER_PATH = "/workspace/OneTrainer"

# Python executable path in the container
PYTHON_CMD = "python"

# Default ranks and epochs - can be overridden in the request
DEFAULT_LORA_RANKS = [64]
DEFAULT_EPOCHS = [200]

# TCP server configuration
TCP_HOST = '0.0.0.0'  # Listen on all interfaces
TCP_PORT = 8080       # Port for TCP connection
MAX_CLIENTS = 5       # Maximum number of concurrent clients

# Ensure required directories exist
os.makedirs("/workspace/input", exist_ok=True)
os.makedirs("/workspace/output", exist_ok=True)
os.makedirs("/workspace/models", exist_ok=True)
os.makedirs("/workspace/temp", exist_ok=True)

# Path to the base configuration file
BASE_CONFIG_PATH = "/workspace/lora_training_config_example.json"

def generate_captions(directory, caption_prefix):
    """Generate captions for images in the directory"""
    subprocess.run([
        PYTHON_CMD,
        os.path.join(ONETRAINER_PATH, "scripts", "generate_captions.py"),
        "--model", "BLIP",
        "--sample-dir", directory,
        "--caption-prefix", f"{caption_prefix} ",
        "--initial-caption", "a woman ",
        "--mode", "fill"
    ], cwd=ONETRAINER_PATH)

def generate_masks(directory):
    """Generate masks for images in the directory"""
    subprocess.run([
        PYTHON_CMD,
        os.path.join(ONETRAINER_PATH, "scripts", "generate_masks.py"),
        "--model", "CLIPSEG",
        "--sample-dir", directory,
        "--add-prompt", "Head",
        "--threshold", "0.3",
        "--smooth-pixels", "5",
        "--expand-pixels", "10",
        "--alpha", "1"
    ], cwd=ONETRAINER_PATH)

def generate_loras(config_path):
    """Train a LoRA model with the given configuration"""
    subprocess.run([
        PYTHON_CMD,
        os.path.join(ONETRAINER_PATH, "scripts", "train.py"),
        "--config-path", config_path
    ], cwd=ONETRAINER_PATH)

def save_base_config(config_data):
    """Save the provided base configuration or use default if not provided"""
    if not config_data:
        # Create a default configuration
        default_config = {
            "project_name": "lora_training",
            "lora_rank": 64,
            "epochs": 200,
            "save_every_n_epochs": 50,
            "network_category": "lora",
            "network_dim": 64,
            "lr": 1e-4,
            "unet_lr": 1e-4,
            "text_encoder_lr": 1e-5,
            "resolution": 512,
            "batch_size": 2,
            "clip_skip": 2,
            "logging_dir": "/workspace/logs",
            "save_precision": "fp16",
            "save_model_as": "safetensors",
            "train_data_dir": "/workspace/input",
            "output_dir": "/workspace/output",
            "output_model_destination": "/workspace/models/model.safetensors",
            "concepts": [
                {
                    "path": "/workspace/input",
                    "instance_prompt": "a photo of a concept"
                }
            ]
        }
        config_data = default_config
        
    with open(BASE_CONFIG_PATH, 'w') as f:
        json.dump(config_data, indent=4, fp=f)

def receive_files_handler(job):
    """Endpoint to handle file receiving via runpodctl"""
    job_input = job["input"]
    
    # If a base configuration is provided, save it
    if "base_config" in job_input:
        save_base_config(job_input["base_config"])
        
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
            # Change directory to input directory
            os.chdir("/workspace/input")
            
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
        "message": f"File transfer initiated with code: {one_time_code}. Transfer will continue in background."
    }

def train_handler(job):
    """Endpoint to start training"""
    job_input = job["input"]
    
    # Get model configuration from the request
    model_config = job_input.get("model", {})
    directory = model_config.get("directory", "/workspace/input")
    trigger_word = model_config.get("trigger_word", "concept")
    prefix_save_as = model_config.get("prefix_save_as", "model-")
    
    # Get training parameters
    lora_ranks = job_input.get("lora_ranks", DEFAULT_LORA_RANKS)
    epochs = job_input.get("epochs", DEFAULT_EPOCHS)
    
    # Verify the directory exists and contains files
    if not os.path.exists(directory):
        return {"status": "error", "message": f"Training directory not found: {directory}"}
    
    files = os.listdir(directory)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not image_files:
        return {"status": "error", "message": f"No image files found in {directory}. Make sure the extraction was successful."}
    
    print(f"Found {len(image_files)} image files in {directory}")
    
    # Load the base configuration
    try:
        with open(BASE_CONFIG_PATH) as f:
            base_config = json.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": "Base configuration file not found. Upload it first."}
    
    # Process training
    results = []
    
    # Generate captions
    generate_captions(directory, trigger_word)
    results.append("Captions generated")
    
    # Generate masks
    generate_masks(directory)
    results.append("Masks generated")
    
    # Train LoRA models
    trained_models = []
    training_combinations = list(itertools.product(lora_ranks, epochs))
    for lora_rank, epoch in training_combinations:
        # Create a copy of the base configuration
        config = copy.deepcopy(base_config)
        config["lora_rank"] = lora_rank
        config["epochs"] = epoch
        
        # Update paths in the configuration
        if "concepts" in config and len(config["concepts"]) > 0:
            config["concepts"][0]["path"] = directory
        
        # Set output paths
        config["save_filename_prefix"] = f"{prefix_save_as}{lora_rank}lr-{epoch}e-"
        output_model_path = f"/workspace/models/{prefix_save_as}{epoch}e-{lora_rank}lr.safetensors"
        config["output_model_destination"] = output_model_path
        
        # Disable TensorBoard to avoid the missing executable error
        config["use_tensorboard"] = False
        
        # Save the temporary configuration
        temp_config_path = f"/workspace/temp/config_{prefix_save_as.rstrip('-')}_{lora_rank}_{epoch}.json"
        with open(temp_config_path, "w") as temp_file:
            json.dump(config, temp_file, indent=4)
        
        # Train the model
        generate_loras(temp_config_path)
        
        # Clean up
        os.remove(temp_config_path)
        
        results.append(f"LoRA trained: {output_model_path}")
        trained_models.append(output_model_path)
    
    return {
        "status": "success", 
        "results": results,
        "models": trained_models,
        "message": "Training completed. Use 'send_model' endpoint to get a one-time code for downloading each model."
    }

def send_model_handler(job):
    """Endpoint to send a trained model via runpodctl"""
    job_input = job["input"]
    
    # Get the model path
    model_path = job_input.get("model_path")
    if not model_path:
        # If no specific model, prepare to send all models
        return {
            "status": "error",
            "message": "Please specify which model to send using the 'model_path' parameter"
        }
    
    # Make sure the model exists
    if not os.path.exists(model_path):
        return {
            "status": "error",
            "message": f"Model not found: {model_path}"
        }
    
    try:
        # Change to the directory containing the model
        os.chdir(os.path.dirname(model_path))
        
        # Run runpodctl send
        model_filename = os.path.basename(model_path)
        cmd = ["runpodctl", "send", model_filename]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # Extract the one-time code from the output
        # Output example: "Code is: 8338-galileo-collect-fidel"
        output_lines = result.stdout.strip().split('\n')
        code_line = [line for line in output_lines if "Code is:" in line]
        
        if not code_line:
            return {
                "status": "error",
                "message": "Failed to get one-time code",
                "details": result.stdout
            }
        
        one_time_code = code_line[0].split("Code is:")[1].strip()
        
        return {
            "status": "success",
            "message": f"Ready to send {model_filename}",
            "one_time_code": one_time_code,
            "instructions": f"On your local machine run: runpodctl receive {one_time_code}"
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error sending model: {e}",
            "details": e.stderr if hasattr(e, 'stderr') else str(e)
        }

def list_models_handler(job):
    """Endpoint to list all trained models"""
    models_dir = "/workspace/models"
    
    if not os.path.exists(models_dir):
        return {
            "status": "error",
            "message": "Models directory not found"
        }
    
    # List all .safetensors files in the models directory
    models = []
    for file in os.listdir(models_dir):
        if file.endswith(".safetensors"):
            model_path = os.path.join(models_dir, file)
            models.append({
                "filename": file,
                "path": model_path,
                "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2)
            })
    
    return {
        "status": "success",
        "models": models,
        "message": "Use 'send_model' endpoint with a 'model_path' parameter to get a one-time code for downloading"
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
    "train": train_handler,
    "send_model": send_model_handler,
    "list_models": list_models_handler,
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