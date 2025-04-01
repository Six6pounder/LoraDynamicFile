import runpod
import os
import json
import copy
import itertools
import subprocess
import shutil
from tqdm import tqdm
import time

# Base path for OneTrainer in the container
ONETRAINER_PATH = "/workspace/OneTrainer"

# Python executable path in the container
PYTHON_CMD = "python"

# Default ranks and epochs - can be overridden in the request
DEFAULT_LORA_RANKS = [64]
DEFAULT_EPOCHS = [200]

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
    
    # Execute runpodctl to receive the files
    try:
        # Change directory to input directory
        os.chdir("/workspace/input")
        
        # Run the receive command
        cmd = ["runpodctl", "receive", one_time_code]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        return {
            "status": "success",
            "message": "Files received successfully",
            "details": result.stdout
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error receiving files: {e}",
            "details": e.stderr if hasattr(e, 'stderr') else str(e)
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

# Map endpoints to handlers
handlers = {
    "receive_files": receive_files_handler,
    "train": train_handler,
    "send_model": send_model_handler,
    "list_models": list_models_handler,
    "shutdown": shutdown_handler
}

# RunPod serverless handler
def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "")
    
    if action in handlers:
        return handlers[action](job)
    else:
        return {
            "status": "error",
            "message": f"Unknown action: {action}. Available actions: {list(handlers.keys())}"
        }

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler}) 