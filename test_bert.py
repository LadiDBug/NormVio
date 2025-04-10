import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm

# Imposta dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Caricamento del Tokenizer e del Modello BERT pre-addestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# 2. Dati di esempio
texts = [
    "I love machine learning!",
    "Deep learning is the future of AI.",
    "I hate waiting in long lines.",
    "I enjoy learning new things every day.",
    "I do not like traffic jams.",
    "I love this movie, it was fantastic!",
    "This is the worst book I have ever read.",
    "Amazing performance and great plot.",
    "Terrible acting and bad script."
]
labels = [1, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positivo, 0 = negativo

# 3. Tokenizzazione dei dati
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

# 4. Creazione dataset e DataLoader
input_ids = inputs['input_ids']
attention_masks = inputs['attention_mask']
labels_tensor = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Ottimizzatore e loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# 6. Funzione di training
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training..."):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        # Passaggio in avanti
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Calcolo della perdita e retropropagazione
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # Predizione del batch (puoi aggiungere questa parte per ogni batch)
        for i in range(len(input_ids)):
            predicted_class = predictions[i].item()
            print(f"Text: {tokenizer.decode(input_ids[i], skip_special_tokens=True)}")
            print(f"Predicted class: {predicted_class} | True label: {labels[i].item()}")
            print("-" * 50)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# 7. Esecuzione del training
num_epochs = 15  # Impostato a 2 per il test, puoi aumentarlo a 10 o più per allenamenti veri
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    avg_loss, accuracy = train(model, train_dataloader, optimizer, criterion)
    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
