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
import logging # Import aggiunto

# --- Configurazione Logging ---
# Configura un logger di base. L'output andrà nei log standard del container RunPod.
# Puoi personalizzare ulteriormente il formato o l'output se necessario.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)

# --- Costanti ---
COMFYUI_PATH = "/workspace/ComfyUI"
# Python executable path in the container (Verifica se è necessario o se UV_CMD è sufficiente)
PYTHON_CMD = "python"
UV_CMD = "/workspace/ComfyUI/venv/bin/python" # Usato per avviare ComfyUI

# TCP server configuration
TCP_HOST = '0.0.0.0'  # Ascolta su tutte le interfacce
TCP_PORT = 8080       # Porta per la connessione TCP
MAX_CLIENTS = 20      # Massimo numero di client concorrenti

# --- Funzione Background per Ricezione File ---
def receive_files_in_background(one_time_code):
    """
    Task eseguito in background per ricevere file tramite runpodctl e processarli.
    Include logging dettagliato e gestione degli errori.
    """
    thread_name = threading.current_thread().name
    logging.info(f"[{thread_name}] Starting file receive task with code: {one_time_code}")
    receive_cmd = ["runpodctl", "receive", one_time_code]
    # runpodctl di solito scarica nella directory di lavoro corrente
    current_dir = os.getcwd()
    logging.info(f"[{thread_name}] Expected download directory: {current_dir}")

    try:
        logging.info(f"[{thread_name}] Executing command: {' '.join(receive_cmd)}")
        # Esegui il comando runpodctl. Timeout impostato a 1800 secondi (30 minuti).
        # check=True solleverà CalledProcessError se il comando fallisce.
        subprocess.run(receive_cmd, check=True, timeout=1800)
        logging.info(f"[{thread_name}] runpodctl receive command completed successfully for code: {one_time_code}")

        # --- Inizio Post-Processing ---
        logging.info(f"[{thread_name}] Starting post-processing in directory: {current_dir}")

        # Sposta model.safetensors se esiste nella directory loras
        source_model_path = os.path.join(current_dir, "model.safetensors")
        loras_dir = os.path.join(COMFYUI_PATH, "models", "loras")
        target_model_path = os.path.join(loras_dir, "model.safetensors") # Mantiene il nome originale

        if os.path.exists(source_model_path):
            logging.info(f"[{thread_name}] Found '{source_model_path}'. Attempting to move to '{loras_dir}'")
            try:
                # Assicurati che la directory di destinazione esista
                os.makedirs(loras_dir, exist_ok=True)
                shutil.move(source_model_path, target_model_path)
                logging.info(f"[{thread_name}] Successfully moved '{source_model_path}' to '{target_model_path}'")
            except Exception as move_error:
                # Logga qualsiasi errore durante lo spostamento
                logging.error(f"[{thread_name}] Failed to move '{source_model_path}': {move_error}", exc_info=True)
        else:
            logging.warning(f"[{thread_name}] File '{source_model_path}' not found after runpodctl finished.")

        logging.info(f"[{thread_name}] Post-processing finished for code: {one_time_code}")
        # --- Fine Post-Processing ---

    except subprocess.CalledProcessError as e:
        # Errore specifico se runpodctl fallisce (exit code != 0)
        logging.error(f"[{thread_name}] runpodctl command failed with return code {e.returncode} for code {one_time_code}.", exc_info=True)
        # Potresti voler loggare e.output o e.stderr se catturati con capture_output=True
    except subprocess.TimeoutExpired:
        # Errore specifico se runpodctl supera il timeout
        logging.error(f"[{thread_name}] runpodctl command timed out after 1800 seconds for code {one_time_code}.", exc_info=True)
    except Exception as e:
        # Cattura qualsiasi altra eccezione imprevista durante l'intero processo
        logging.error(f"[{thread_name}] An unexpected error occurred during receive/processing for code {one_time_code}: {e}", exc_info=True) # exc_info=True aggiunge il traceback al log


# --- Handler Endpoints ---

def receive_files_handler(job):
    """Endpoint per gestire la ricezione file tramite runpodctl (avvia task in background)."""
    job_input = job["input"]
    one_time_code = job_input.get("one_time_code")
    if not one_time_code:
        logging.warning("Receive files request received without one_time_code.")
        return {
            "status": "error",
            "message": "No one-time code provided for file transfer"
        }

    logging.info(f"Received request to initiate file transfer with code: {one_time_code}")
    # Avvia la funzione di ricezione e processamento in un thread separato
    # Passa il codice come argomento e assegna un nome al thread per il logging
    thread = threading.Thread(
        target=receive_files_in_background,
        args=(one_time_code,),
        name=f"Receive-{one_time_code[:8]}" # Usa parte del codice per nome thread
    )
    thread.daemon = True  # Permette al programma principale di terminare anche se il thread è attivo
    thread.start()
    logging.info(f"Background thread '{thread.name}' started for file transfer.")

    # Ritorna immediatamente per non bloccare la richiesta HTTP
    return {
        "status": "success",
        "message": f"File transfer initiated with code: {one_time_code}. Processing continues in background. Check pod logs for progress and errors."
    }

def shutdown_handler(job):
    """Endpoint per richiedere lo spegnimento del pod (informativo)."""
    logging.info("Received shutdown request (handler provides info only).")
    return {
        "status": "success",
        "message": "To shutdown the pod, use the RunPod API or web interface. This endpoint is informational."
    }

def check_upload_handler(job):
    """Controlla lo stato dei file nella directory di input."""
    input_dir = "/workspace/input"
    logging.info(f"Checking uploaded files in directory: {input_dir}")

    if not os.path.exists(input_dir):
        logging.warning(f"Input directory '{input_dir}' does not exist.")
        return {
            "status": "error",
            "message": "Input directory does not exist"
        }

    try:
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            relative_path = os.path.relpath(root, input_dir)
            if relative_path == ".":
                relative_path = "" # Evita "./" nel percorso

            for file in files:
                file_path = os.path.join(relative_path, file)
                all_files.append(file_path)

        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
        num_files = len(all_files)
        num_images = len(image_files)
        logging.info(f"Found {num_files} total files, including {num_images} image files.")

        return {
            "status": "success",
            "message": f"Found {num_files} files in total, including {num_images} image files.",
            "files": all_files,
            "image_count": num_images
        }
    except Exception as e:
        logging.error(f"Error checking files in '{input_dir}': {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error checking files: {str(e)}"
        }

def check_status_handler(job):
    """Controlla se ComfyUI è pronto a ricevere comandi tramite l'endpoint /system_stats."""
    comfyui_url = "http://127.0.0.1:8188/system_stats" # Usa 127.0.0.1 perchè nello stesso container
    logging.info(f"Checking ComfyUI status at {comfyui_url}")
    try:
        response = requests.get(comfyui_url, timeout=5) # Aggiungi un timeout
        if response.status_code == 200:
            logging.info("ComfyUI is ready.")
            return {
                "status": "success",
                "ready": True,
                "message": "ComfyUI is ready to receive commands"
            }
        else:
            logging.warning(f"ComfyUI status check failed with status code: {response.status_code}")
            return {
                "status": "error",
                "ready": False,
                "message": f"ComfyUI returned status code {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking ComfyUI status: {e}", exc_info=True)
        return {
            "status": "error",
            "ready": False,
            "message": f"Error connecting to ComfyUI: {str(e)}"
        }
    except Exception as e: # Cattura altre eccezioni impreviste
        logging.error(f"Unexpected error checking ComfyUI status: {e}", exc_info=True)
        return {
            "status": "error",
            "ready": False,
            "message": f"Unexpected error checking ComfyUI status: {str(e)}"
        }


def check_lora_handler(job):
    """Controlla se uno specifico modello LoRA esiste nella directory corretta."""
    job_input = job["input"]
    lora_name = job_input.get("lora_name")

    if not lora_name:
        logging.warning("Check LoRA request received without lora_name.")
        return {"status": "error", "message": "No LoRA model name provided"}

    lora_dir = os.path.join(COMFYUI_PATH, "models", "loras")
    logging.info(f"Checking for LoRA model '{lora_name}' in directory: {lora_dir}")

    if not os.path.isdir(lora_dir): # Controlla se è una directory
        logging.error(f"LoRA directory '{lora_dir}' does not exist or is not a directory.")
        return {"status": "error", "message": "LoRA directory does not exist"}

    lora_path = os.path.join(lora_dir, lora_name)
    if os.path.isfile(lora_path): # Controlla se è un file
        try:
            file_size = os.path.getsize(lora_path)
            logging.info(f"LoRA model '{lora_name}' found. Size: {file_size} bytes.")
            return {
                "status": "success",
                "message": f"LoRA model '{lora_name}' found with size {file_size} bytes.",
                "model_found": True,
                "file_path": lora_path,
                "file_size": file_size
            }
        except OSError as e:
             logging.error(f"Error getting size for LoRA file '{lora_path}': {e}", exc_info=True)
             return {
                 "status": "error",
                 "message": f"Found LoRA model '{lora_name}' but failed to get size: {e}",
                 "model_found": True,
                 "file_path": lora_path,
             }
    else:
        logging.warning(f"Exact LoRA model '{lora_name}' not found. Searching for similar names...")
        available_models = []
        try:
            available_models = [f for f in os.listdir(lora_dir) if os.path.isfile(os.path.join(lora_dir, f))]
        except OSError as e:
            logging.error(f"Error listing files in LoRA directory '{lora_dir}': {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"LoRA model '{lora_name}' not found and failed to list directory content.",
                "model_found": False,
            }

        possible_matches = [f for f in available_models if f.lower().startswith(lora_name.lower())]

        if possible_matches:
            logging.info(f"Found {len(possible_matches)} potential matches for '{lora_name}'.")
            match_details = []
            for match in possible_matches:
                 match_path = os.path.join(lora_dir, match)
                 match_size = -1 # Default size if error occurs
                 try:
                    match_size = os.path.getsize(match_path)
                 except OSError:
                    logging.warning(f"Could not get size for potential match '{match_path}'.")
                 match_details.append({"name": match, "size": match_size})

            return {
                "status": "success", # Considerato successo perchè abbiamo trovato alternative
                "message": f"Exact LoRA model '{lora_name}' not found, but found {len(possible_matches)} similar models.",
                "model_found": False, # Il modello esatto non è stato trovato
                "exact_match_found": False,
                "similar_models_found": True,
                "similar_models": match_details
            }
        else:
            logging.warning(f"LoRA model '{lora_name}' not found. No similar models detected.")
            # Limita il numero di modelli disponibili nella risposta per evitare payload troppo grandi
            max_listed_models = 20
            listed_models = available_models[:max_listed_models]
            message = f"LoRA model '{lora_name}' not found."
            if len(available_models) > max_listed_models:
                 message += f" Listing first {max_listed_models} available models."

            return {
                "status": "error", # Errore perchè il modello richiesto non c'è
                "message": message,
                "model_found": False,
                "exact_match_found": False,
                "similar_models_found": False,
                "available_models_sample": listed_models,
                "total_available_models": len(available_models)
            }

# --- Mapping Endpoints -> Handlers ---
handlers = {
    "receive_files": receive_files_handler,
    "shutdown": shutdown_handler,
    "check_upload": check_upload_handler,
    "status": check_status_handler,
    "check_lora": check_lora_handler
}

# --- RunPod HTTP Handler ---
def http_handler(job):
    """Gestore principale per le richieste HTTP serverless di RunPod."""
    job_input = job.get("input", {})
    action = job_input.get("action", "")
    request_id = job.get("id", "unknown") # Ottieni ID richiesta RunPod se disponibile
    logging.info(f"Received HTTP request (ID: {request_id}). Action: '{action}'")

    if action in handlers:
        try:
            result = handlers[action](job)
            log_level = logging.INFO if result.get("status") == "success" else logging.WARNING
            logging.log(log_level, f"Responding to HTTP request (ID: {request_id}, Action: {action}). Status: {result.get('status')}")
            return result
        except Exception as e:
            logging.error(f"Unhandled exception in handler for action '{action}' (ID: {request_id}): {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Internal server error processing action '{action}': {str(e)}"
            }
    else:
        logging.warning(f"Received request (ID: {request_id}) with unknown action: '{action}'")
        available_actions = list(handlers.keys())
        return {
            "status": "error",
            "message": f"Unknown action: {action}. Available actions: {available_actions}"
        }

# --- TCP Server Implementation ---
def handle_client(client_socket, client_address):
    """Gestisce una singola connessione client TCP."""
    addr_str = f"{client_address[0]}:{client_address[1]}"
    thread_name = threading.current_thread().name
    logging.info(f"[{thread_name}] Accepted TCP connection from {addr_str}")
    try:
        data = b""
        client_socket.settimeout(60.0) # Timeout per la ricezione dati
        while True:
            try:
                chunk = client_socket.recv(4096)
                if not chunk:
                    logging.info(f"[{thread_name}] TCP Connection closed by client {addr_str} (empty chunk).")
                    break # Connessione chiusa dal client
                data += chunk
                # Supponiamo che un messaggio completo termini con newline
                if b"\n" in data:
                    break
            except socket.timeout:
                logging.warning(f"[{thread_name}] TCP receive timeout from {addr_str}. Data received so far: {len(data)} bytes.")
                # Puoi decidere se chiudere o attendere ancora
                break # Chiudiamo per ora
            except Exception as recv_e:
                 logging.error(f"[{thread_name}] Error receiving data from {addr_str}: {recv_e}", exc_info=True)
                 data = None # Indica errore
                 break

        if not data:
            logging.warning(f"[{thread_name}] No complete data received or error occurred for {addr_str}.")
            return # Nessun dato o errore di ricezione

        # Processa solo il primo messaggio completo se ne arrivano multipli
        message, _, remainder = data.partition(b'\n')
        if remainder:
             logging.warning(f"[{thread_name}] Received more data than expected after newline from {addr_str}. Processing only first message.")

        try:
            request_str = message.decode('utf-8')
            logging.debug(f"[{thread_name}] Received TCP data from {addr_str}: {request_str[:200]}...") # Logga solo inizio
            job = json.loads(request_str)
            action = job.get("input", {}).get("action", "")
            logging.info(f"[{thread_name}] Processing TCP request from {addr_str}. Action: '{action}'")

            if action in handlers:
                result = handlers[action](job) # Chiama lo stesso handler HTTP
            else:
                available_actions = list(handlers.keys())
                result = {
                    "status": "error",
                    "message": f"Unknown action: {action}. Available actions: {available_actions}"
                }
            response_bytes = json.dumps(result).encode('utf-8') + b"\n"
            client_socket.sendall(response_bytes)
            logging.info(f"[{thread_name}] Sent TCP response to {addr_str}. Status: {result.get('status')}")

        except json.JSONDecodeError:
            logging.error(f"[{thread_name}] Invalid JSON received from {addr_str}.")
            error_resp = {"status": "error", "message": "Invalid JSON request"}
            client_socket.sendall(json.dumps(error_resp).encode('utf-8') + b"\n")
        except Exception as e:
            logging.error(f"[{thread_name}] Error processing TCP request from {addr_str}: {e}", exc_info=True)
            # Invia un errore generico se possibile
            try:
                error_resp = {"status": "error", "message": f"Internal server error: {str(e)}"}
                client_socket.sendall(json.dumps(error_resp).encode('utf-8') + b"\n")
            except Exception as send_e:
                 logging.error(f"[{thread_name}] Failed to send error response to {addr_str}: {send_e}")

    except Exception as e:
        logging.error(f"[{thread_name}] Unhandled error in client handler for {addr_str}: {e}", exc_info=True)
    finally:
        logging.info(f"[{thread_name}] Closing TCP connection for {addr_str}")
        client_socket.close()

def start_tcp_server(stop_event):
    """Avvia il server TCP per ascoltare le connessioni client."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Permette riutilizzo indirizzo

    try:
        server.bind((TCP_HOST, TCP_PORT))
        server.listen(MAX_CLIENTS)
        logging.info(f"TCP Server listening on {TCP_HOST}:{TCP_PORT}")

        # Timeout sul server socket per permettere controllo stop_event
        server.settimeout(1.0)

        client_threads = []
        while not stop_event.is_set():
            try:
                client_socket, client_address = server.accept()
                # Crea e avvia un thread per gestire il nuovo client
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, client_address),
                    name=f"TCPClient-{client_address[0]}-{client_address[1]}"
                )
                client_thread.daemon = True # Permette uscita anche se client attivi
                client_thread.start()
                client_threads.append(client_thread) # Tieni traccia (opzionale)

                # Pulizia opzionale dei thread terminati (non strettamente necessaria con daemon=True)
                client_threads = [t for t in client_threads if t.is_alive()]

            except socket.timeout:
                # Timeout è normale, usato per controllare stop_event
                continue
            except Exception as e:
                # Logga errori durante l'accept, ma continua ad ascoltare se possibile
                if not stop_event.is_set(): # Evita log se stiamo chiudendo
                    logging.error(f"Error accepting TCP connection: {e}", exc_info=True)
                continue

    except Exception as e:
        logging.critical(f"TCP Server failed to start or encountered a critical error: {e}", exc_info=True)
    finally:
        logging.info("TCP Server shutting down.")
        server.close()
        # Opzionale: attendi brevemente che i thread client terminino (se non sono daemon)
        # for t in client_threads:
        #    t.join(timeout=5.0)


# --- Avvio ComfyUI ---
def start_comfyui(stop_event):
    """Avvia il processo ComfyUI in background."""
    comfyui_cmd = [UV_CMD, "main.py", "--listen", "0.0.0.0"]
    process = None
    try:
        logging.info(f"Starting ComfyUI process with command: {' '.join(comfyui_cmd)} in {COMFYUI_PATH}")
        # Usa Popen per non bloccare e poter gestire lo stop
        process = subprocess.Popen(comfyui_cmd, cwd=COMFYUI_PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(f"ComfyUI process started with PID: {process.pid}")

        # Monitora l'output e l'evento di stop
        while not stop_event.is_set() and process.poll() is None:
            # Leggi output non bloccante (opzionale, ma utile per debug)
            try:
                line_stdout = process.stdout.readline()
                if line_stdout:
                    logging.info(f"[ComfyUI stdout] {line_stdout.strip()}")
                line_stderr = process.stderr.readline()
                if line_stderr:
                     logging.warning(f"[ComfyUI stderr] {line_stderr.strip()}") # Logga stderr come warning
            except Exception: # Ignora errori lettura temporanei
                pass
            time.sleep(0.1) # Breve pausa

        if process.poll() is not None:
             logging.error(f"ComfyUI process terminated unexpectedly with code: {process.poll()}.")
             # Logga output rimanente
             stdout, stderr = process.communicate()
             if stdout: logging.info(f"[ComfyUI remaining stdout] {stdout.strip()}")
             if stderr: logging.warning(f"[ComfyUI remaining stderr] {stderr.strip()}")


    except FileNotFoundError:
         logging.critical(f"Error starting ComfyUI: Command '{UV_CMD}' or 'main.py' not found in {COMFYUI_PATH}.", exc_info=True)
    except Exception as e:
        logging.critical(f"Critical error starting or running ComfyUI: {e}", exc_info=True)
    finally:
        if process and process.poll() is None: # Se stiamo uscendo e il processo è ancora vivo
            logging.info(f"Attempting to terminate ComfyUI process (PID: {process.pid})...")
            try:
                process.terminate() # Prova terminazione gentile
                process.wait(timeout=10) # Aspetta un po'
                logging.info("ComfyUI process terminated.")
            except subprocess.TimeoutExpired:
                logging.warning("ComfyUI process did not terminate gracefully, killing...")
                process.kill() # Forza chiusura
                logging.info("ComfyUI process killed.")
            except Exception as term_e:
                logging.error(f"Error trying to terminate ComfyUI process: {term_e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"RunPod Handler starting up in {os.getcwd()}...")

    # Evento per segnalare lo stop ai thread
    stop_event = threading.Event()

    # Funzione per gestire segnali di terminazione (SIGINT, SIGTERM)
    def signal_handler(signum, frame):
        logging.warning(f"Received signal {signal.Signals(signum).name}. Initiating shutdown...")
        stop_event.set() # Segnala ai thread di fermarsi

    signal.signal(signal.SIGINT, signal_handler) # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Segnale di terminazione standard

    # Avvia il server TCP in un thread separato
    tcp_thread = threading.Thread(target=start_tcp_server, args=(stop_event,), name="TCPServerThread")
    tcp_thread.daemon = True
    tcp_thread.start()
    logging.info("TCP server thread started.")

    # Avvia ComfyUI in un thread separato (che a sua volta lancia un subprocess)
    comfyui_thread = threading.Thread(target=start_comfyui, args=(stop_event,), name="ComfyUIThread")
    comfyui_thread.daemon = True # Anche questo daemon per permettere uscita pulita
    comfyui_thread.start()
    logging.info("ComfyUI thread started.")

    # Avvia il gestore RunPod (che di solito è bloccante)
    # N.B.: Se runpod.serverless.start è bloccante, il codice dopo non viene eseguito
    # fino allo spegnimento. Se non è bloccante o usi un framework diverso,
    # potresti aver bisogno del loop `while not stop_event.is_set()` qui.
    logging.info("Starting RunPod serverless handler...")
    try:
        # Assumendo che http_handler sia la funzione che RunPod chiama
        runpod.serverless.start({
            "handler": http_handler
            # Potrebbero esserci altre opzioni di configurazione qui
        })
        # Se .start ritorna, significa che è stato richiesto lo spegnimento o c'è stato un errore
        logging.info("RunPod serverless handler finished.")

    except Exception as e:
        logging.critical(f"RunPod serverless handler failed: {e}", exc_info=True)
    finally:
        # Assicurati che lo stop event sia settato in ogni caso all'uscita
        if not stop_event.is_set():
            logging.info("Initiating shutdown from main block finally clause.")
            stop_event.set()

        # Attendi che i thread principali (TCP, ComfyUI) terminino (opzionale, ma buona pratica)
        logging.info("Waiting for background threads to finish...")
        if tcp_thread.is_alive():
            tcp_thread.join(timeout=5.0)
            if tcp_thread.is_alive(): logging.warning("TCP server thread did not exit cleanly.")
        if comfyui_thread.is_alive():
            comfyui_thread.join(timeout=15.0) # Dagli più tempo per terminare il subprocess
            if comfyui_thread.is_alive(): logging.warning("ComfyUI thread did not exit cleanly.")

        logging.info("Handler shutdown complete.")
        sys.exit(0) # Esci esplicitamente

# Il loop while commentato sotto sarebbe necessario se runpod.serverless.start non fosse bloccante
# try:
#     logging.info("Entering main loop (waiting for stop event)...")
#     while not stop_event.is_set():
#         try:
#             time.sleep(1)
#         except KeyboardInterrupt: # Redondante se signal_handler funziona
#             logging.warning("\nKeyboardInterrupt received, shutting down...")
#             break
# except Exception as e:
#     logging.critical(f"Main loop error: {e}", exc_info=True)
# finally:
#     if not stop_event.is_set():
#         stop_event.set()
#     # ... (attendi thread come sopra) ...
#     logging.info("Handler shutdown complete.")