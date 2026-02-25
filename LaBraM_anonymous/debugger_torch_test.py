import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready for PyTorch!")
    
    # Get the current GPU device
    device = torch.device("cuda:0")  # You can specify the GPU index if multiple GPUs are available.
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Get total and free memory on the GPU
    total_memory = torch.cuda.get_device_properties(device).total_memory  # Total memory in bytes
    reserved_memory = torch.cuda.memory_reserved(device)  # Reserved memory in bytes
    allocated_memory = torch.cuda.memory_allocated(device)  # Allocated memory in bytes
    free_memory = reserved_memory - allocated_memory  # Free memory in bytes
    
    print(f"Total GPU Memory: {total_memory / 1e9:.2f} GB")
    print(f"Reserved Memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Allocated Memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Free Memory: {free_memory / 1e9:.2f} GB")
else:
    print("CUDA is not available. Using CPU instead.")
