import time
start_time = time.time()
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import hashlib
import numpy as np
from pandas import read_csv


def hash_image(file_path,difficulty):
    if(difficulty <= 0):
        return None
    try:
        with open(file_path, 'rb') as f:
            nonce = 0
            prefix = '0' * difficulty
            while(True):
                image_data = f.read()
                hash_value = hashlib.sha256((str(image_data) + str(nonce)).encode()).hexdigest()
                nonce+=1
                if hash_value.startswith(prefix):
                    break
            return hash_value
    except FileNotFoundError:
        print("File not found.")
        return None



# Load the pre-trained AlexNet model
model = torchvision.models.alexnet(weights='AlexNet_Weights.DEFAULT')
model.eval()  # Set the model to evaluation mode

# Define transformations to be applied to the input image
transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def infer(image_path):
    # Load the image
    # image_path = "dog.jpg"
    # image_path = "perturbed_image.jpg"

    #Puzzle solving
    # difficulty = 2
    target_time = 5
    with open("difficulty.txt") as f:
        data = [line.strip() for line in f.readlines()]
    difficulty = int(data[0])

    print(difficulty)



    hash_value = hash_image(image_path,difficulty)
    print(hash_value)
    image = Image.open(image_path)

    # Apply transformations to the image
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_image)

    # Get predicted class
    predicted_class = output[0]#torch.argmax(output).item()

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(top5_catid)
    # Load class labels
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # Print the predicted class label
    print("Predicted class:", labels[top5_catid[0]])


    #adjust difficulty for next iteration
    total_execution_time = time.time() - start_time
    print("--- %s seconds ---" % (total_execution_time))
    print("--- %s Joules ---" % (total_execution_time*15.0))
    if(total_execution_time<target_time):
        difficulty+=1
        
    else:
        difficulty-=(difficulty>1)
        

    print("New difficulty = " +str (difficulty))


    f = open("difficulty.txt", "w")
    f.writelines([str(difficulty)])
    f.close()