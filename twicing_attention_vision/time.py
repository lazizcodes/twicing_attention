import torch
import time
import statistics
import timm
import models

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

model_name = 'deit_tiny_patch16_224' 
# model_name = 'deit_sharp_tiny_patch16_224' 
model = timm.create_model(model_name, pretrained=False)

print('Load the saved weights...')
checkpoint = torch.load('/sharpformer/models/deit_checkpoint.pth', map_location=device)
# checkpoint = torch.load('/sharpformer/mattention2-imagenet-master/checkpoints/boostattn/checkpoint.pth', map_location=device)
state_dict = checkpoint["model"]
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Move model to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a random input tensor with the appropriate size (e.g., batch_size=1, channels=3, height=224, width=224)
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# Warm-up to make sure the GPU is ready
for _ in range(1000):
    _ = model(input_tensor)

total = 0

sample_count = 1000
samples = []

for _ in range(sample_count):
    torch.cuda.synchronize()  # Synchronize GPU if using CUDA
    start_time = time.time()
    # Run the forward pass
    _ = model(input_tensor)
    torch.cuda.synchronize()  # Synchronize GPU if using CUDA
    end_time = time.time()
    current = (end_time - start_time) * 1000
    total += current
    samples.append(current)


compute_speed_ms_per_it = total / sample_count
print(f"Compute Speed: {compute_speed_ms_per_it:.2f} ms/it")
print('Variance', statistics.variance(samples), 'mean', statistics.mean(samples))
print(f'min {min(samples)}, max {max(samples)}')
