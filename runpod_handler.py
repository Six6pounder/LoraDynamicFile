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

# Flag to indicate server readiness
SERVER_READY = False

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

    # Clean up output directories before starting
    try:
        folders_to_clean = [
            "/workspace/temp_bundle",
            "/workspace/input",
            "/workspace/temp",
            "/workspace/logs",
            "/workspace/OneTrainer/workspace",
            "/workspace/OneTrainer/workspace-cache"
        ]

        # Special handling for models directory to preserve base directory and its contents
        models_dir = "/workspace/models"
        if os.path.exists(models_dir):
            print(f"Cleaning models directory while preserving base folder...")
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                
                # Skip the base directory and its contents
                if item == "base" and os.path.isdir(item_path):
                    print(f"Preserving base directory: {item_path}")
                    continue
                
                # Remove other files and directories
                if os.path.isfile(item_path):
                    # if item is inside base folder, don't delete it
                    if "base" in item_path:
                        print(f"Skipping base folder: {item_path}")
                        continue
                    else:
                        os.remove(item_path)
                        print(f"Removed file: {item_path}")

        # Clean up the other directories
        for folder in folders_to_clean:
            if os.path.exists(folder):
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"Removed directory: {item_path}")
                    else:
                        os.remove(item_path)
                        print(f"Removed file: {item_path}")
        
    except Exception as e:
        print(f"Warning: Error cleaning up directories: {e}")
        # Continue even if cleanup fails

    # If a base configuration is provided, save it
    if "base_config" in job_input:
        save_base_config(job_input["base_config"])
    else:
        # Clean up existing configuration file before using default
        if os.path.exists(BASE_CONFIG_PATH):
            try:
                os.remove(BASE_CONFIG_PATH)
                print(f"Removed existing configuration file: {BASE_CONFIG_PATH}")
            except Exception as e:
                print(f"Error removing configuration file: {e}")
        save_base_config(None)  # This will create a default config
        
    # Get the one-time code for receiving files
    one_time_code = job_input.get("one_time_code")
    if not one_time_code:
        return {
            "status": "error",
            "message": "No one-time code provided for file transfer"
        }
    
    # Get training parameters for later use after file upload
    model_config = job_input.get("model", {})
    directory = model_config.get("directory", "/workspace/input")
    trigger_word = model_config.get("trigger_word", "concept")
    prefix_save_as = model_config.get("prefix_save_as", "model-")
    lora_ranks = job_input.get("lora_ranks", DEFAULT_LORA_RANKS)
    epochs = job_input.get("epochs", DEFAULT_EPOCHS)
    auto_train = job_input.get("auto_train", True)
    
    # Start a background thread to receive the files
    def receive_files_in_background():
        try:
            # Change directory to input directory
            os.chdir("/workspace/")
            
            # Run the receive command
            cmd = ["runpodctl", "receive", one_time_code]
            subprocess.run(cmd, check=True, timeout=600)
            print(f"Files received successfully with code: {one_time_code}")
            
            # Find and extract the tar.gz file
            extracted = False
            for file in os.listdir("."):
                if file.endswith(".tar.gz"):
                    print(f"Extracting archive: {file}")
                    subprocess.run(["tar", "-xzf", file], check=True)
                    
                    # Remove the tar file after extraction
                    os.remove(file)
                    print(f"Extracted contents and removed archive: {file}")
                    extracted = True
                    
                    # If a directory named 'temp_bundle' exists, move its contents up
                    if os.path.exists("temp_bundle"):
                        for item in os.listdir("temp_bundle"):
                            # Salta i file che iniziano con punto (nascosti)
                            if item.startswith('.'):
                                print(f"Skipping hidden file/directory: {item}")
                                continue
                                
                            src_path = os.path.join("temp_bundle", item)
                            if os.path.exists(item):
                                if os.path.isdir(item):
                                    shutil.rmtree(item)
                                else:
                                    os.remove(item)
                            shutil.move(src_path, "/workspace/input")
                        os.rmdir("temp_bundle")
                        print("Moved contents from temp_bundle to input directory")
                    break
            
            # If extraction completed successfully and auto_train is enabled, start training
            if extracted and auto_train:
                print("File upload and extraction completed. Starting training automatically...")
                
                # Verify the directory exists and contains image files
                if not os.path.exists(directory):
                    print(f"Training directory not found: {directory}")
                    return
                
                files = os.listdir(directory)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                
                if not image_files:
                    print(f"No image files found in {directory}. Cannot start training.")
                    return
                
                print(f"Found {len(image_files)} image files in {directory}")
                
                # Load the base configuration
                try:
                    with open(BASE_CONFIG_PATH) as f:
                        base_config = json.load(f)
                except FileNotFoundError:
                    print("Base configuration file not found. Cannot start training.")
                    return
                
                # Process training (using the same code as in train_handler)
                try:
                    # Process training
                    results = []
                    
                    # Generate captions
                    print("Starting caption generation...")
                    generate_captions(directory, trigger_word)
                    results.append("Captions generated")
                    
                    # Generate masks
                    print("Starting mask generation...")
                    generate_masks(directory)
                    results.append("Masks generated")
                    
                    # Train LoRA models
                    trained_models = []
                    training_combinations = list(itertools.product(lora_ranks, epochs))
                    for lora_rank, epoch in training_combinations:
                        print(f"Starting LoRA training for rank {lora_rank} and {epoch} epochs...")
                        # Create a copy of the base configuration
                        config = copy.deepcopy(base_config)
                        config["lora_rank"] = lora_rank
                        config["epochs"] = epoch
                        
                        # Update paths in the configuration
                        if "concepts" in config and len(config["concepts"]) > 0:
                            config["concepts"][0]["path"] = directory
                        
                        # Set output paths
                        config["save_filename_prefix"] = f"{prefix_save_as}{lora_rank}lr-{epoch}e-"
                        output_model_path = f"/workspace/models/{prefix_save_as.rstrip('-')}.safetensors"
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
                        
                    print("All training completed successfully!")
                    print(f"Results: {results}")
                    print(f"Trained models: {trained_models}")
                    
                except Exception as e:
                    print(f"Error during automatic training: {e}")
            
        except Exception as e:
            print(f"Error receiving or extracting files: {e}")
    
    # Start background thread
    thread = threading.Thread(target=receive_files_in_background)
    thread.daemon = True
    thread.start()
    
    # Return immediately to avoid blocking
    return {
        "status": "success",
        "message": f"Trasferimento file avviato con codice: {one_time_code}. Il trasferimento continuerà in background e l'addestramento inizierà automaticamente dopo il completamento del caricamento." if job_input.get("auto_train", True) else f"Trasferimento file avviato con codice: {one_time_code}. Il trasferimento continuerà in background. I file precedenti verranno eliminati."
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
    
    # Check if an upload is in progress by looking for .tar.gz files or runpodctl receive processes
    upload_in_progress = False
    
    # Check for tar.gz files in the input directory which would indicate an extraction in progress
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith(".tar.gz"):
                upload_in_progress = True
                print(f"Found tar.gz file in progress: {file}")
                break
    
    # Check for runpodctl receive processes
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        if "runpodctl receive" in result.stdout:
            upload_in_progress = True
            print("Found runpodctl receive process running")
    except Exception as e:
        print(f"Error checking for runpodctl processes: {e}")
    
    if upload_in_progress:
        print("Upload in progress, waiting before starting training...")
        return {
            "status": "waiting",
            "message": "Un caricamento file è attualmente in corso. Riprova tra qualche minuto quando il caricamento sarà completato."
        }
    
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
    
    # Define the training function to run in background
    def run_training_in_background():
        try:
            # Process training
            results = []
            
            # Generate captions
            print("Starting caption generation...")
            generate_captions(directory, trigger_word)
            results.append("Captions generated")
            
            # Generate masks
            print("Starting mask generation...")
            generate_masks(directory)
            results.append("Masks generated")
            
            # Train LoRA models
            trained_models = []
            training_combinations = list(itertools.product(lora_ranks, epochs))
            for lora_rank, epoch in training_combinations:
                print(f"Starting LoRA training for rank {lora_rank} and {epoch} epochs...")
                # Create a copy of the base configuration
                config = copy.deepcopy(base_config)
                config["lora_rank"] = lora_rank
                config["epochs"] = epoch
                
                # Update paths in the configuration
                if "concepts" in config and len(config["concepts"]) > 0:
                    config["concepts"][0]["path"] = directory
                
                # Set output paths
                config["save_filename_prefix"] = f"{prefix_save_as}{lora_rank}lr-{epoch}e-"
                output_model_path = f"/workspace/models/{prefix_save_as.rstrip('-')}.safetensors"
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
                
            print("All training completed successfully!")
            print(f"Results: {results}")
            print(f"Trained models: {trained_models}")
            
        except Exception as e:
            print(f"Error during training: {e}")
    
    # Start the training in a background thread
    training_thread = threading.Thread(target=run_training_in_background)
    training_thread.daemon = True
    training_thread.start()
    
    # Immediately return a response to avoid TCP timeout
    return {
        "status": "started", 
        "message": "Training has started and is running in the background. Use 'list_models' to check for completed models."
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
        
        # Get the model filename
        model_filename = os.path.basename(model_path)
        print(f"Preparing to send model: {model_filename}")
        
        # First run runpodctl send to get the one-time code
        cmd = ["runpodctl", "send", model_filename]
        print(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read output lines until we find the one-time code
        one_time_code = None
        output_lines = []
        
        # Set a maximum time to wait for the code
        start_time = time.time()
        max_wait_time = 30  # seconds
        
        while time.time() - start_time < max_wait_time:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            line = line.strip()
            output_lines.append(line)
            print(f"Send output: {line}")
            
            # Check various formats of how the code might be presented
            if "Code is:" in line:
                one_time_code = line.split("Code is:")[1].strip()
                print(f"Found one-time code (format 1): {one_time_code}")
                break
            elif "Code:" in line:
                one_time_code = line.split("Code:")[1].strip()
                print(f"Found one-time code (format 2): {one_time_code}")
                break
            elif "-" in line and len(line.split()) <= 2:
                # This looks like just the code by itself (e.g., 4240-chemist-saddle-rabbit-6)
                one_time_code = line
                print(f"Found one-time code (format 3): {one_time_code}")
                break
        
        # If we didn't find the code in the standard output, check if it was the last line printed
        if not one_time_code and output_lines:
            last_line = output_lines[-1]
            if "-" in last_line and len(last_line.split("-")) >= 2:
                one_time_code = last_line
                print(f"Using last line as one-time code: {one_time_code}")
        
        # Manually set the one-time code if we saw it in the logs but couldn't parse it
        if not one_time_code and job_input.get("manual_code"):
            one_time_code = job_input.get("manual_code")
            print(f"Using manually provided one-time code: {one_time_code}")
            
        # If we still don't have a code, check if the process wrote anything to stderr
        if not one_time_code:
            stderr_output = ""
            for line in process.stderr:
                stderr_output += line
                print(f"Error output: {line.strip()}")
            
            if stderr_output:
                return {
                    "status": "error",
                    "message": "Failed to get one-time code",
                    "details": f"Error output: {stderr_output}",
                    "stdout": "\n".join(output_lines)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to get one-time code",
                    "details": "Could not parse one-time code from output",
                    "stdout": "\n".join(output_lines)
                }
        
        # Start a background thread to handle the rest of the file sending
        def send_file_in_background():
            try:
                # Let the process continue running to completion
                stdout, stderr = process.communicate()
                exit_code = process.wait()
                
                if exit_code != 0:
                    print(f"Error in runpodctl send background process: {stderr}")
                else:
                    print(f"Model {model_filename} was successfully prepared for sending with code: {one_time_code}")
                    if stdout:
                        print(f"Full output: {stdout}")
            except Exception as e:
                print(f"Error in background file sending: {e}")
        
        # Start the background thread
        thread = threading.Thread(target=send_file_in_background)
        thread.daemon = True
        thread.start()
        
        # Return immediately with the one-time code
        return {
            "status": "success",
            "message": f"Model {model_filename} is ready to be downloaded",
            "one_time_code": one_time_code,
            "instructions": f"On your local machine run: runpodctl receive {one_time_code}"
        }
        
    except Exception as e:
        print(f"Exception in send_model_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error preparing model for sending: {e}",
            "details": str(e)
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
    in_progress_models = []
    
    for file in os.listdir(models_dir):
        if file.endswith(".safetensors"):
            model_path = os.path.join(models_dir, file)
            
            # Check if the file is still being written to
            # by measuring the file size, waiting a moment, and checking again
            try:
                initial_size = os.path.getsize(model_path)
                time.sleep(1)  # Wait a short time
                current_size = os.path.getsize(model_path)
                
                # If size changed, file is still being written
                if current_size != initial_size:
                    in_progress_models.append({
                        "filename": file,
                        "path": model_path,
                        "size_mb": round(current_size / (1024 * 1024), 2),
                        "status": "in_progress"
                    })
                    continue
                
                # Extra check - wait a bit longer for larger files
                if current_size > 100 * 1024 * 1024:  # If larger than 100MB
                    time.sleep(2)  # Wait a bit more
                    final_check_size = os.path.getsize(model_path)
                    if final_check_size != current_size:
                        in_progress_models.append({
                            "filename": file,
                            "path": model_path,
                            "size_mb": round(final_check_size / (1024 * 1024), 2),
                            "status": "in_progress"
                        })
                        continue
                
                # If we got here, file size is stable - model is ready
                models.append({
                    "filename": file,
                    "path": model_path,
                    "size_mb": round(current_size / (1024 * 1024), 2),
                    "status": "complete"
                })
            except Exception as e:
                # If there's an error accessing the file, it might be in use
                print(f"Error checking model file {file}: {e}")
                in_progress_models.append({
                    "filename": file,
                    "path": model_path,
                    "size_mb": 0,
                    "status": "error",
                    "error": str(e)
                })
    
    return {
        "status": "success",
        "models": models,
        "in_progress_models": in_progress_models,
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
        # Check if there are any runpodctl receive processes running
        upload_in_progress = False
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            if "runpodctl receive" in result.stdout:
                upload_in_progress = True
                print("Found runpodctl receive process running")
        except Exception as e:
            print(f"Error checking for runpodctl processes: {e}")
        
        # Check for tar.gz files in the input directory which would indicate an extraction in progress
        tar_files = []
        if os.path.exists(input_dir):
            tar_files = [f for f in os.listdir(input_dir) if f.endswith('.tar.gz')]
            if tar_files:
                upload_in_progress = True
                print(f"Found tar files waiting for extraction: {tar_files}")
        
        # Get all files in the input directory
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            relative_path = os.path.relpath(root, input_dir)
            if relative_path == ".":
                relative_path = ""
            
            for file in files:
                # Skip .tar.gz files in the count of "real" files
                if file.endswith('.tar.gz'):
                    continue
                file_path = os.path.join(relative_path, file)
                all_files.append(file_path)
        
        # Count image files
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        status_message = f"Found {len(all_files)} files in total, including {len(image_files)} image files"
        if upload_in_progress:
            status_message = "Upload in progress. " + status_message
        
        return {
            "status": "success",
            "message": status_message,
            "files": all_files,
            "image_count": len(image_files),
            "upload_in_progress": upload_in_progress,
            "tar_files": tar_files
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking files: {str(e)}"
        }

def check_status_handler(job):
    """Endpoint to check if server is ready to receive commands"""
    return {
        "status": "success",
        "ready": SERVER_READY,
        "message": "Server is ready to receive commands" if SERVER_READY else "Server is initializing",
        "available_actions": list(handlers.keys())
    }

# Map endpoints to handlers
handlers = {
    "receive_files": receive_files_handler,
    "train": train_handler,
    "send_model": send_model_handler,
    "list_models": list_models_handler,
    "shutdown": shutdown_handler,
    "check_upload": check_upload_handler,
    "status": check_status_handler
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
    global SERVER_READY
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((TCP_HOST, TCP_PORT))
        server.listen(MAX_CLIENTS)
        print(f"TCP Server listening on {TCP_HOST}:{TCP_PORT}")
        
        # Server is now ready to accept connections
        SERVER_READY = True
        print("Server is ready to receive commands")
        
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
        SERVER_READY = False
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
        SERVER_READY = False
        stop_event.set() 