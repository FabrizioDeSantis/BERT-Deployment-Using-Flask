from transformers import AutoModel, BertModel
import torch
from torch import nn
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pytorch_lightning as pl
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup

model_ckpt = "model-bert-cpi.ckpt"
BERT_MODEL="dbmdz/bert-base-italian-uncased"
THRESHOLD = 0.50
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

mlb = MultiLabelBinarizer()

yt=mlb.fit_transform([[1.0], [2.1], [2.2], [2.3], [3.1], [3.2], [3.3], [4.1], [4.2], [4.3], [4.4], [5.1], [5.2], [6.1], [6.2], [6.3], [7.0], [8.1], [8.2]])

class MultiLabelClassifier(pl.LightningModule):
    def __init__(self, n_classes=19, n_epochs=10, steps_per_epoch=None, learning_rate=3e-5):
        super().__init__()
        self.bert=BertModel.from_pretrained(BERT_MODEL, return_dict=True) # recupero modello preaddestrato
        self.classifier=nn.Linear(self.bert.config.hidden_size, n_classes) # applico un classificatore lineare
        self.steps_per_epoch=steps_per_epoch
        self.n_epochs=n_epochs
        self.learning_rate=learning_rate
        self.criterion=nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output=self.classifier(output.pooler_output) # questo ritorna il classification token dopo averlo processato attraverso un linear layer con funzione di attivazione tanh
        output=torch.sigmoid(output)
        loss=0
        if labels is not None:
            loss=self.criterion(output, labels)

        return loss, output

    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #outputs = self(input_ids,attention_mask)
        #loss = self.criterion(outputs, labels)

        loss, outputs = self(input_ids, attention_mask, labels)

        self.log('train_loss',loss , prog_bar=True,logger=True)
        
        return {"loss" :loss, "predictions":outputs, "labels": labels }


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #outputs = self(input_ids,attention_mask)
        #loss = self.criterion(outputs,labels)

        loss, outputs = self(input_ids, attention_mask, labels)

        self.log('val_loss',loss , prog_bar=True,logger=True)
        
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #outputs = self(input_ids,attention_mask)
        #loss = self.criterion(outputs,labels)

        loss, outputs = self(input_ids, attention_mask, labels)

        self.log('test_loss',loss , prog_bar=True,logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.learning_rate)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='Task-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    question = request.form["inference-text"]
    text_enc = tokenizer.encode_plus(
                    question.lower(),
                    None,
                    add_special_tokens=True,
                    max_length=512,
                    padding = 'max_length',
                    return_token_type_ids= False,
                    return_attention_mask= True,
                    truncation=True,
                    return_tensors = 'pt'      
    )
    _, outputs = model(text_enc['input_ids'], text_enc['attention_mask'])
    pred_out = outputs[0].detach().numpy()
    preds = [(pred > THRESHOLD) for pred in pred_out ]
    preds = np.asarray(preds)
    new_preds = preds.reshape(1,-1).astype(int)
    pred_tags = mlb.inverse_transform(new_preds)
    string=""
    for i in range(0, len(pred_tags[0])):
        splitted = str(pred_tags[0][i]).split(".")
        if i==len(pred_tags[0])-1:
            if splitted[0]=="0":
                string = string + "Titolo " + splitted[0] + " "
            else:
                string = string + "Titolo " + splitted[0] + " Capo " + splitted[1] + " "
        else:
            if splitted[0]=="0":
                string = string + "Titolo " + splitted[0] + ", "
            else:
                string = string + "Titolo " + splitted[0] + " Capo " + splitted[1] + ", "
    return render_template('index.html', previous_text=f"{question}", prediction_text=f'Capi predetti: {string}', massima_text=f'Testo inserito: {question}', isDisplay=True, maintainText=True)

@app.route("/")
def hello_world():
    return render_template('index.html', isDisplay=False, maintainText=False)
if __name__ == "__main__":
    model_path = checkpoint_callback.best_model_path
    model = MultiLabelClassifier.load_from_checkpoint("Task-epoch=04-val_loss=0.05.ckpt")
    app.run()