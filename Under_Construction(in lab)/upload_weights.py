import ujson as json

# Load the model weights from the JSON file
with open("model_weights.json", "r") as f:
    model_weights = json.load(f)

# Convert lists back into usable format
coefs = model_weights["coefs"]
intercepts = model_weights["intercepts"]

# Print basic information
print("Model weights loaded successfully!")
print("Number of layers:", len(coefs))
print("Shape of first layer weights:", len(coefs[0]), "neurons")
