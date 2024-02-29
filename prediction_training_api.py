import pickle
import os
import json
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
from datetime import datetime
from transformers import Trainer, TrainingArguments
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math

analyser = SentimentIntensityAnalyzer()

def cal_score(sentiment_list):
    score=0
    labels_score = {'very negative': -0.41, 'negative': -0.4, 'positive': 0.3, 'neutral':-0.2, 'very positive': 0.5}
    
    for sentiment in sentiment_list:
         score+=labels_score[sentiment]
    
    if len(sentiment_list)==1:
         return score
    else:
        max_score = 0.5*(len(sentiment_list))
        average_score = round(score/max_score, 2)
        return average_score

def most_common(sentiment_list):
    sentiment_counts = Counter(sentiment_list)
    most_common_sentiment, _ = sentiment_counts.most_common(1)[0]

    return most_common_sentiment

f1_scores = []

def compute_metrics(pred):
    global f1_scores
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels,preds,average='weighted')
    f1_scores.append(f1)
    return {'f1': f1}
     
model_ckpt = 'microsoft/MiniLM-L12-H384-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_text(texts, max_length=100):
    # Use the tokenizer to encode the texts with truncation and padding
    encoded_texts = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'  # 'pt' for PyTorch, 'tf' for TensorFlow
    )

    return encoded_texts

def classify_sentiment(x):
    if x >= 0.5:
        return "very positive"
    elif 0.3 <= x < 0.5:
        return "positive"
    elif -0.2 <= x < 0.3:
        return "neutral"
    elif -0.4 <= x < -0.2:    
        return "negative"
    else:
        return "very negative"
    
def create_labels(comments):
     
    predictions = [classify_sentiment(analyser.polarity_scores(comment)['compound']) for comment in comments]
    class_labels = ['negative','neutral','positive','very negative','very positive']
    labels = [class_labels.index(pred) for pred in predictions]
    print(labels)
    return labels

class Sentiment_Dataset(Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self,idx):
        item = {key:(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        
        return len(self.labels)

#train model with new data until good accuracy is achieved, then use it to perform sentiment analysis on user comments


def train_model(train_dataset,eval_dataset, model):#retrain model with modififed training, it will not be used until good accuracy is achieved, apply condition
        
        output_dir = './twitter_miniLM_combined'
        training_args = TrainingArguments(
        output_dir= output_dir,         
        num_train_epochs=5,            
        per_device_train_batch_size=5,  
        per_device_eval_batch_size=5,   
        learning_rate = 5e-5,           
        weight_decay = 0.1,
        evaluation_strategy='epoch',            
        logging_dir='./logs',            
    )
        modified_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=5)
        trainer = Trainer(
            model= modified_model,                         
            args=training_args,                 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStopping(monitor='val_loss', patience=3)]      
        )

        # Start training
        trainer.train()

        print(f1_scores)
        f1 = f1_scores[-1]

        if f1 >= 0.85:
            print("The model has been modified")
            return modified_model
        
        else:
             return model

model_app = Flask(__name__)

@model_app.route('/training', methods =['POST'])

def training(): #filename should be renamed to username; class of comments 

    username = request.form.get('username')

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    input_file = request.files['file']

    if input_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        users_data = json.load(input_file)
        thresh =  users_data['comments']

    except:
        return "Something went wrong while reading this file, select appropriate file"

    datetime_current = datetime.now()
    date_formatted = datetime_current.strftime('%m/%d/%y')
    time_formatted = datetime_current.strftime('%I:%M %p')

    users_data['date'] = date_formatted 
    users_data['time'] = time_formatted 

    # Save the modified json_data back to the file
    with open(input_file.filename, 'w') as file:
        json.dump(users_data, file, indent=2)
    
    threshold = 50  #this threshold is to decide the minimum number of comments needed for training the model; obviously it needs to be higher
   
    if len(thresh) >= threshold:
        
        print("threshold satisfied")
        comments = [comment for comment in users_data['comments']]
        labels = create_labels(comments)
        trainX, evalX, trainY, evalY = train_test_split(comments, labels, test_size=0.2, random_state = 100)
        training_encoded = tokenize_text(trainX)
        eval_encoded = tokenize_text(evalX)
        
        #train the model
        train_dataset = Sentiment_Dataset(training_encoded,trainY)
        eval_dataset = Sentiment_Dataset(eval_encoded,evalY)
    
        with open('sentiment_model_3.pkl', 'rb') as file:
                model = pickle.load(file)

        retrained_model = train_model(train_dataset,eval_dataset,model)

        weeks_score = users_data['weeks_score']

        if not weeks_score['avg_score']: 
                
            classified_labels = [classify_sentiment(analyser.polarity_scores(comment)['compound']) for comment in users_data['comments']]
            users_data['labels'].extend(classified_labels)
            avg_score_initial = cal_score(classified_labels)
            weeks_score['avg_score'].append(avg_score_initial)

        with open(f'{username}.json', 'w') as file:
            json.dump(users_data, file, indent=2)

        with open('retrained_model.pkl', 'wb') as file:
            pickle.dump(retrained_model, file)

        return "We have trained model successfully"
    
    else:
        return "Data not sufficient for training"

   
@model_app.route('/prediction', methods =['POST'])

def prediction(): 

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    new_comments_file = request.files['file']

    if new_comments_file.filename == '':
        return jsonify({'error': 'No selected file'})


    username = request.form.get('username')
    
    filepath = os.path.join(os.getcwd(),f'{username}.json')
    print(filepath)
    if os.path.exists(filepath):
        
        num_weeks = request.form.get('weeks')

        if num_weeks is not None:

            with open(f'{username}.json','r') as file:
                original_data = json.load(file)
            
            weeks_score = original_data["weeks_score"]
            print(weeks_score['avg_score'])
            avg_score = weeks_score['avg_score'][-1]

            try:
                json_data = json.load(new_comments_file)
                total_comments = json_data.get('comments', {})

            except:
                return "Something went wrong while reading this file, select appropriate file"

            print(original_data['comments'])

            original_data['comments'].extend(total_comments)

            print(original_data)

            num_comments = len(total_comments) 
            print(num_comments)
           
            num_weeks = int(num_weeks)
            min_comments = num_weeks*7

            if num_comments >= min_comments:

                print("Weekly threshold satisfied")
                #transform data
                tokenized_text = tokenize_text(total_comments)

                with open('retrained_model.pkl', 'rb') as file:
                        model = pickle.load(file)

                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                input_ids = tokenized_text['input_ids'].to(device)
                attention_mask = tokenized_text['attention_mask'].to(device)


                class_labels = ['negative','neutral','positive','very negative','very positive']

                # Make predictions

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
                logits = outputs.logits
                prediction_indxs = torch.argmax(logits, dim=-1)
                prediction_indxs = prediction_indxs.cpu().numpy()
                print(prediction_indxs)
                prediction_labels = [class_labels[i] for i in prediction_indxs]
                print(prediction_labels)

                original_data['labels'].extend(prediction_labels)

                cur_score = cal_score(prediction_labels)

                print(cur_score)
                print(avg_score)

                average_sentiment = classify_sentiment(avg_score)
                current_sentiment = classify_sentiment(cur_score)

                overall = cal_score(original_data['labels'])
                print(overall)
                overall_sentiment = classify_sentiment(overall)
                weeks_score['avg_score'].append(overall)
                weeks_score['weeks'].append(num_weeks)

                with open(f'{username}.json','w') as file:
                    json.dump(original_data, file, indent=2)
                
                if current_sentiment == average_sentiment:
                    if cur_score < avg_score:
                        return jsonify({'overall_sentiment':  overall_sentiment, 'average_sentiment': average_sentiment, 'current_sentiment': current_sentiment,'status':"There has been no significant change, however there has been a slight decline"})
                    elif cur_score > avg_score:
                        return jsonify({'overall_sentiment':  overall_sentiment, 'average_sentiment': average_sentiment, 'current_sentiment': current_sentiment,'status':"There has been no significant change, however there has been a slight improvement"}) 
                    else:
                        return "there has no change"
                    
                if cur_score < avg_score:
                    return jsonify({'overall_sentiment':  overall_sentiment, 'average_sentiment': average_sentiment, 'current_sentiment': current_sentiment,'status':"user is going in a negative direction"})
                if cur_score > avg_score:
                    return jsonify({'overall_sentiment':  overall_sentiment, 'average_sentiment': average_sentiment, 'current_sentiment': current_sentiment,'status':"user is going in a positive direction"})
                 
            else:
                return jsonify({"error":"Comments made are not sufficient for improvement analysis"})
        else:
            return jsonify({'error':'Enter number of weeks'}),400
    else:
        return jsonify({'error':'Please enter valid username'}),400
    
        
if __name__ == '__main__':
        model_app.run(debug=True, host = '0.0.0.0')