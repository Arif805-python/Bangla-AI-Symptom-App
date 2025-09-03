from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os

# --- মডেল লোড করার যুক্তি ---
# এই কোডটি সার্ভার চালু হওয়ার সময় মাত্র একবার চলবে
model_name = "ArifulRussell5200/Bangla-Symptom-Checker"
# Render-এর পার্সিস্টেন্ট ডিস্কের জন্য পাথ
model_dir = "/var/data/model" 

# পাইপলাইন তৈরির ফাংশন
def create_pipeline():
    # চেক করা হচ্ছে মডেলটি ডিস্কে আগে থেকেই ডাউনলোড করা আছে কিনা
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print(f"{model_dir}-এ মডেল ডাউনলোড করা হচ্ছে...")
        os.makedirs(model_dir, exist_ok=True)
        # মডেল ডাউনলোড এবং সেভ করা
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        print("মডেল ডিস্কে আগে থেকেই আছে।")
    
    # ডিস্ক থেকে মডেল লোড করে পাইপলাইন রিটার্ন করা
    return pipeline('text-classification', model=model_dir)

# অ্যাপ্লিকেশন চালু হওয়ার সময় পাইপলাইন তৈরি করা
symptom_checker = create_pipeline()
print("AI পাইপলাইন প্রস্তুত।")

# --- Flask অ্যাপ ---
app = Flask(__name__)

@app.route('/')
def home():
    return "AI Symptom Checker is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        symptom_text = data['symptom']
        
        # প্রেডিকশন পাওয়ার জন্য পাইপলাইন ব্যবহার করা
        result = symptom_checker(symptom_text)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})