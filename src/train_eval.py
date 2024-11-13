import torch
from model import initialize_model, setup_optimizer_scheduler, train_epoch, evaluate
from data import load_data, create_data_loader, tokenizer
from src.setup import device

# Load and preprocess data
dataset = load_data()
train_loader = create_data_loader(dataset['train'], tokenizer)
val_loader = create_data_loader(dataset['validation'], tokenizer)
test_loader = create_data_loader(dataset['test'], tokenizer)

# Initialize the model
num_labels = len(dataset['train'].features['intent'].names)
model = initialize_model(num_labels)
model.to(device)

# Set up optimizer and scheduler
optimizer, scheduler = setup_optimizer_scheduler(model, train_loader)

# Training and validation loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_accuracy, val_report = evaluate(model, val_loader, device)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the best model
    torch.save(model.state_dict(), f"model_weights_{epoch}.pth")

# Final evaluation on the test set
test_loss, test_accuracy, test_report = evaluate(model, test_loader, device)
print("Test Classification Report:", test_report)
