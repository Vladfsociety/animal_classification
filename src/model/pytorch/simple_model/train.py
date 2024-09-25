import time
import torch
from torch.amp import GradScaler, autocast
from .build import CNN
from ..load_data import get_train_test_datasets
from ..report import save_metrics


def train():
	num_classes = 10	
	learning_rate = 0.001
	weight_decay = 0.001
	num_epochs = 30

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_data, test_data = get_train_test_datasets()

	model = CNN(num_classes=num_classes).to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	scaler = GradScaler('cuda')

	history = {
    	"loss": [],
    	"accuracy": [],
    	"val_loss": [],
    	"val_accuracy": [],
	}

	start_time = time.time()

	for epoch in range(num_epochs):
	    model.train()
	    running_loss = 0.0
	    correct_preds = 0
	    total_preds = 0
	    batches_num = 0
	    for i, (images, labels) in enumerate(train_data):
	        images, labels = images.to(device), labels.to(device)
	        optimizer.zero_grad()

	        with autocast('cuda'):
	            outputs = model(images)
	            loss = criterion(outputs, labels)
	        scaler.scale(loss).backward()
	        scaler.step(optimizer)
	        scaler.update()
	        
	        running_loss += loss.item()

	        probabilities = torch.nn.functional.softmax(outputs, dim=1)
	        
	        _, predicted = torch.max(probabilities.data, 1)
	        total_preds += labels.size(0)
	        correct_preds += (predicted == labels).sum().item()
	        batches_num += 1

	    model.eval()
	    
	    val_loss = 0.0
	    val_correct_preds = 0
	    val_total_preds = 0
	    with torch.no_grad():
	        for images, labels in test_data:
	            images, labels = images.to(device), labels.to(device)
	            outputs = model(images)
	            loss = criterion(outputs, labels)
	            val_loss += loss.item()

	            probabilities = torch.nn.functional.softmax(outputs, dim=1)
	            _, predicted = torch.max(probabilities.data, 1)
	            val_total_preds += labels.size(0)
	            val_correct_preds += (predicted == labels).sum().item()
	    
	    avg_loss = running_loss / batches_num
	    accuracy = (correct_preds / total_preds) * 100
	    avg_val_loss = val_loss / len(test_data)
	    val_accuracy = (val_correct_preds / val_total_preds) * 100
	    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, Test Loss: {avg_val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%')

	    history["loss"].append(avg_loss)
	    history["accuracy"].append(accuracy)
	    history["val_loss"].append(avg_val_loss)
	    history["val_accuracy"].append(val_accuracy)
	            
	print("--- Model train time: %s seconds ---" % round(time.time() - start_time, 4))

	save_metrics(
		history,
		num_epochs,
		"Training and Validation Accuracy (PyTorch simple model)",
		"reports/pytorch/simple_model/accuracy.jpg",
		"Training and Validation Loss (PyTorch simple model)",
		"reports/pytorch/simple_model/loss.jpg"
	)

	torch.save(model.state_dict(), 'models/pytorch/simple_model.pth')

if __name__ == '__main__':
    train()