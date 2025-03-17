import random
from datetime import datetime
import math
import os
import re

import multiprocessing
import socket

import cv2
import mediapipe as mp
import time
import sys
import json
from llama_cpp import Llama
import openai
from openai import OpenAI
import numpy as np

import torch
import psutil
import csv
import contextlib
import platform

LlmService = "llm"
llmContextSize=2048#I like 512 for small CPU tasks. Common values: 2048, 4096, 8192, or even higher (e.g., GPT-4-Turbo supports >32K tokens).
llmBatchSize=1424
maxTokens=1424#Limits the model's output
llmIsWorkingFlag=False
startTime=time.time()
LlmKey=''
llmWaitTime=0
totalTime=0
prompt='test1'

csvFilePath='C:/Users/velas/OneDrive - KISP Inc/Product_Desc_Label.csv'
semaphoreNumber=1
semaphore = multiprocessing.Semaphore(semaphoreNumber)
#controls number of times the query gets used as input
numberOfPromptFeedback=4
#Controls randomness in the model’s output. smaller is less random
minCreativity=0.2
maxCreativity=1.1#max 1.5

#chatbot configuration
#for prompt engineering
randomWords1=['Change The wording','Rewrite this','Enhance', 'Decrease the amount of words',
                      'Improve','Be Creative','In one sentence','Be brief']
randomWords2=['summarize','explain','be concise','use descriptors for','describe']
#for System Role
randomAiPhrases = ['You are a helpful chatbot.', 'You are trying to create a text for a search bar',
                            'You are a witty chatbot that tries to add humor', 'You are a helpful assistant',
                            'You are a concise AI that responds with short and direct answers.',
                            'you are an interior designer explaining different features',
                            'You are an experienced furniture seller that helps find the right object']


print(f"Platform Machine: {platform.machine()}")

print(f'torch version: {torch.__version__}')

if torch.cuda.is_available():
    print(f"CUDA is available! 🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Running on CPU. ❌")


def get_max_batch_size(context_size=4096, token_memory=1500):
    if torch.cuda.is_available():
        # GPU Available: Use VRAM
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory  # Total VRAM (bytes)
        reserved_memory = torch.cuda.memory_reserved(device)  # Reserved VRAM
        available_memory = total_memory - reserved_memory  # Free VRAM
        overhead_ratio = 0.3  # Estimate 30% overhead for model weights, activations, etc.
    else:
        # CPU Mode: Use system RAM
        total_memory = psutil.virtual_memory().total  # Total system RAM (bytes)
        available_memory = psutil.virtual_memory().available  # Available RAM
        overhead_ratio = 0.5  # Estimate 50% overhead for CPU (less memory efficiency)

    # Calculate usable memory
    usable_memory = available_memory * (1 - overhead_ratio)

    # Compute maximum batch size
    max_batch_size = usable_memory // (context_size * token_memory)

    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Detected device: {device_type}")
    print(f"Estimated max batch size: {int(max_batch_size)}")

    return int(max_batch_size)

llmBatchSize=get_max_batch_size()

def generate_text(
    llm,
    prompt="What are the five common emotions?",
    max_tokens=24,
    temperature=0.2,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text

def getAllGuffModels(directory):
    # Create a list to store file paths and sizes
    files_with_sizes = []
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gguf'):
                file_path = os.path.join(root, file)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                files_with_sizes.append((file_path, file_size_mb))
    # Sort the list by file size (smallest first)
    sorted_files = sorted(files_with_sizes, key=lambda x: x[1])
    # Extract only the file paths
    sorted_file_paths = [file_path for file_path, _ in sorted_files]
    return sorted_file_paths

def loadLLM(thePath,contextSize=512,batchSize=126):
    sortedPaths=getAllGuffModels("./")
    print(f"sortedPaths: {sortedPaths}")
    directory_path = os.path.dirname(csvFilePath)
    newFilename = f"llama_cpp_prompts_{datetime.now().strftime('%Y-%m-%d')}.log"
    logFileName = os.path.join(directory_path, newFilename)

    log_file = logFileName
    with open(log_file, "w") as f:
        with contextlib.redirect_stderr(f):
            theLLM = Llama(model_path=sortedPaths[0], n_ctx=contextSize, n_batch=batchSize,use_gpu=True,n_gpu_layers=-1)#n_gpu_layers=32
    return theLLM, log_file

def generate_prompt_from_template(input, loop=1):
    if loop==1:
        chat_prompt_template = f"""<|im_start|>system
            You are a helpful chatbot.<|im_end|>
            <|im_start|>user
            {input}<|im_end|>"""
    else:
        chat_prompt_template = f"""<|im_start|>system
                    {random.choice(randomAiPhrases)}<|im_end|>
                    <|im_start|>user
                    {input}<|im_end|>"""
    return chat_prompt_template

def is_connected():
    try:
        # Connect to a well-known host to check internet access
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False


def getLLM(csvFilePath, csvLineNumber, prompt,theInformation):
    """Calls LLM model to generate response and writes results to CSV."""
    with semaphore:  # Limits concurrent execution to 4 processes
        print("calling local LLM: ")
        llm, logFile = loadLLM("zephyr-7b-beta.Q4_K_M.gguf", llmContextSize, llmBatchSize)
        originalPrompt=prompt
        i = 1
        times_restarted=0
        while i <= numberOfPromptFeedback:
            print(f"Processing prompt: {prompt[:200]}...")  # Truncate for readability
            if i>1:
                prompt=f"{random.choice(randomWords1)} and {random.choice(randomWords2)} the following: "+prompt2

            prompt2 = generate_prompt_from_template(prompt,loop=i)
            result = generate_text(llm, prompt2,temperature=random.uniform(minCreativity,maxCreativity), max_tokens=maxTokens)
            result = re.sub(r"<.*?>", "", result)  # Remove HTML-like tags
            result = re.sub(r"[^a-zA-Z0-9\s]", "", result)  # Remove special characters except spaces
            result = result.replace("\n", " ").replace("\r", "").strip()  # Remove newlines and carriage returns
            prompt2 = result
            if re.search(r"[a-zA-Z]", prompt2) is None:
                i = 1  # Reset loop index when no letters exist
                times_restarted+=1
                prompt2=originalPrompt
            else:
                i += 1  # Increment normally if the result is valid
            if times_restarted>5:
                times_restarted=0

                restartText=f"Restarted 5 times! Line:{csvLineNumber}"
                print(restartText)  # Truncate for readability
                with open(logFile, mode="a", newline="") as file:
                    restartText=restartText+f", prompt: {prompt2}"
                    writerLog = csv.writer(file)
                    writerLog.writerow([restartText])
                i = numberOfPromptFeedback

        result = result+f'###{theInformation}'
        # Save result to a CSV file
        filename, extension = os.path.splitext(os.path.basename(csvFilePath))
        directory_path = os.path.dirname(csvFilePath)
        newFilename = f"{filename}_prompts_{datetime.now().strftime('%Y-%m-%d')}{extension}"
        saveCSVPath = os.path.join(directory_path, newFilename)

        with open(saveCSVPath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([result])

        print(f"Saved result to: {saveCSVPath}")

def worker(row,totalrows, theSemaphore):
    """Worker function to process a row and release semaphore when done."""
    try:
        rowContentSplitTab = row[0].strip().split('\t')
        manufacturer = rowContentSplitTab[0] if len(rowContentSplitTab) > 0 else ""
        vendorCode = rowContentSplitTab[1] if len(rowContentSplitTab) > 1 else ""
        CatalogCode = rowContentSplitTab[2] if len(rowContentSplitTab) > 2 else ""
        Descr = rowContentSplitTab[3] if len(rowContentSplitTab) > 3 else ""

        label = " ".join(row[1:-2]) if len(row) > 3 else ""
        AICategory = row[-1] if len(row) > 1 else ""
        AISubCategory = row[-2] if len(row) > 2 else ""

        splitInformation=f'{manufacturer},{CatalogCode}, {vendorCode}, {Descr}, {label}, {AICategory}, {AISubCategory}'
        prompt = (f"Provide a simple description of an object based on its attributes. "
                  f"Use the following extra information:{splitInformation}. "
                  f"Avoid directly stating {splitInformation}.")

        getLLM(csvFilePath, totalrows, prompt,splitInformation)  # Simulate LLM processing

    except Exception as e:
        print(f"Error processing row: {e}")
    finally:
        theSemaphore.release()  # Release semaphore when process finishes

def mainLoop():
    totalAllTime=time.time()
    """Reads CSV and processes rows 4 at a time, waiting for completion before continuing."""
    with open(csvFilePath, "r", encoding="utf-8") as file:
        row_count = sum(1 for _ in file) - 1  # Subtract 1 for the header

    with open(csvFilePath, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        total_rows = 0
        semaphore = multiprocessing.Semaphore(semaphoreNumber)  # Limit concurrency
        processes = []

        for row in reader:
            total_rows += 1
            semaphore.acquire()  # Lock one process slot

            process = multiprocessing.Process(target=worker, args=(row,total_rows, semaphore))
            processes.append(process)
            process.start()

            print(f"Started process {process.pid} for row {total_rows}/{row_count}: {str(round((total_rows/row_count)*100,2))}% ")

            # If 4 processes are running, wait for them to finish before starting more
            if len(processes) >= semaphoreNumber:
                for p in processes:
                    p.join()  # Wait for each process in the batch to finish
                    print(f"Process {p.pid} finished.")

                processes.clear()  # Reset process list for the next batch

        # Ensure remaining processes finish
        for p in processes:
            p.join()
            print(f"Final process {p.pid} finished.")

    totalAllTime = time.time()-totalAllTime
    hours, remainder = divmod(totalAllTime, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    print(f"Total rows processed: {total_rows}, took {formatted_time}")

# Execute:
if __name__ == '__main__':
    multiprocessing.freeze_support()
    mainLoop()