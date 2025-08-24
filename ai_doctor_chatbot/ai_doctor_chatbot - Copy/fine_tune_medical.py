# fine_tune_medical.py (for old Transformers <= 4.2)

import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# ------------------ Step 1: Synthetic Dataset ------------------
symptoms = [
    "headache", "fever", "sore throat", "stomach pain", "cough", "dizziness",
    "back pain", "tiredness", "cold", "chest pain", "allergy", "vomiting",
    "diarrhea", "ear pain", "toothache", "knee pain", "anxiety", "stress",
    "skin rash", "muscle pain", "shortness of breath", "eye pain", "nausea",
    "insomnia", "high blood pressure", "low blood pressure", "heart palpitations",
    "indigestion", "constipation", "acid reflux", "sneezing", "runny nose",
    "joint pain", "swelling in legs", "loss of appetite", "weight loss",
    "weight gain", "memory problems", "blurred vision", "hearing loss",
    "itchy skin", "burning sensation in eyes", "hair loss", "chills",
    "night sweats", "loss of taste", "loss of smell", "difficulty swallowing",
    "frequent urination", "burning while urinating", "blood in urine",
    "irregular heartbeat", "swollen glands", "weakness", "dry mouth",
    "nosebleeds", "sensitivity to light", "difficulty breathing", "hoarseness",
    "stiff neck", "loss of balance", "tremors", "seizures", "confusion",
    "panic attacks", "restlessness", "irritability", "mood swings"
]

advice = [
    "Please take rest and drink plenty of fluids.",
    "Consider taking paracetamol or ibuprofen as prescribed.",
    "Try warm salt water gargles for throat pain.",
    "Eat a light, healthy diet and avoid oily food.",
    "Stay hydrated and monitor your symptoms regularly.",
    "If symptoms persist for more than 3 days, consult a doctor.",
    "Practice deep breathing and relaxation techniques.",
    "Use an ice pack or warm compress for pain relief.",
    "Take proper sleep and maintain a balanced diet.",
    "Avoid dust, smoke, and allergens in your environment.",
    "Do gentle stretching or light physical activity.",
    "Take prescribed antibiotics only if recommended by a doctor.",
    "Limit screen time and take regular breaks.",
    "Avoid caffeine and alcohol until you feel better.",
    "Wear comfortable clothing and avoid tight garments.",
    "Practice meditation or yoga to reduce stress.",
    "Monitor your blood pressure and sugar levels if needed.",
    "Maintain personal hygiene and wash hands frequently.",
    "Use over-the-counter cough syrup if necessary.",
    "Avoid sharing utensils or close contact if contagious.",
    "Schedule a follow-up checkup if symptoms continue.",
    "Do not self-medicate with strong drugs without advice.",
    "Apply moisturizer or soothing lotion for skin issues.",
    "Keep your room well-ventilated and avoid dampness.",
    "Take probiotics or fiber-rich food for digestion problems."
]


def format_conversation(patient, doctor):
    return f"### Patient:\nI have {patient}.\n\n### Doctor:\n{doctor}\n"

# Generate ~100 samples
def build_dataset(n_samples=100):
    data = []
    for _ in range(n_samples):
        symptom = random.choice(symptoms)
        doctor_advice = random.choice(advice)
        data.append({"text": format_conversation(symptom, doctor_advice)})
    return Dataset.from_list(data)

# ------------------ Step 2: Data Split ------------------
dataset = build_dataset(100)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

# ------------------ Step 3: Model & Tokenizer ------------------
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# ------------------ Step 4: Tokenization ------------------
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ------------------ Step 5: Training ------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_steps=10,
    save_total_limit=1,
    learning_rate=5e-5,
    warmup_steps=10,
    weight_decay=0.01,
    fp16=torch.cuda.is_available()
)

# ⚠️ Old versions may not support evaluation_strategy
# Trainer will still evaluate if eval_dataset is given
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)

print("\nStarting Training...\n")
trainer.train()
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)

# Save model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("\nModel saved to ./fine_tuned_model")

# ------------------ Step 6: Test Generation ------------------
def chat_with_model(user_input):
    input_text = f"### Patient:\nI have {user_input}.\n\n### Doctor:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example test
print("\nGenerated Test Response:")
print(chat_with_model("fever"))