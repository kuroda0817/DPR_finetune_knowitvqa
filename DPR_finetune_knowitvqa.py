import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# import sys
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW

batch_size = 128

class KQADataset:
    def __init__(self, kfile):
        self.k = pd.read_table(kfile)["question"].values
        self.q = pd.read_table(kfile)["reason"].values
        self.a1 = pd.read_table(kfile)["answer1"].values
        self.a2 = pd.read_table(kfile)["answer2"].values
        self.a3 = pd.read_table(kfile)["answer3"].values
        self.a4 = pd.read_table(kfile)["answer4"].values
        

    def __getitem__(self, index):
        return (self.k[index], self.q[index], self.a1[index], self.a2[index],self.a3[index],self.a4[index])

    def __len__(self):
        return self.q.shape[0]
    
dataset_train = KQADataset("knowit_data/knowit_data_train.csv")
dataset_valid = KQADataset("knowit_data/knowit_data_val.csv")
dataset_test = KQADataset("knowit_data/knowit_data_test.csv")

train_loader = torch.utils.data.DataLoader(
      dataset=dataset_train,
      batch_size=128,
      shuffle=True,
      num_workers=2,
      drop_last=True)

test_loader = torch.utils.data.DataLoader(
      dataset=dataset_test,
      batch_size=128,
      shuffle=False,
      num_workers=2,
      drop_last=True)

valid_loader = torch.utils.data.DataLoader(
      dataset=dataset_valid,
      batch_size=128,
      shuffle=False,
      num_workers=2,
      drop_last=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
question_model = torch.nn.DataParallel(DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")).to(device)
context_model = torch.nn.DataParallel(DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")).to(device)
context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")


print("intialized models/tokenizers")


print("created dataset")

def traversal_for_question_context(query, knowledge, answer1, answer2, answer3, answer4):

    question_tokens=question_tokenizer([x + y for (x, y) in zip(list(query), list(answer1))],padding=True,return_tensors='pt').to(device)["input_ids"]
    question_emb1=question_model(question_tokens).pooler_output.reshape(1,128,768)
    question_tokens=question_tokenizer([x + y for (x, y) in zip(list(query), list(answer2))],padding=True,return_tensors='pt').to(device)["input_ids"]
    question_emb2=question_model(question_tokens).pooler_output.reshape(1,128,768)
    question_tokens=question_tokenizer([x + y for (x, y) in zip(list(query), list(answer3))],padding=True,return_tensors='pt').to(device)["input_ids"]
    question_emb3=question_model(question_tokens).pooler_output.reshape(1,128,768)
    question_tokens=question_tokenizer([x + y for (x, y) in zip(list(query), list(answer4))],padding=True,return_tensors='pt').to(device)["input_ids"]
    question_emb4=question_model(question_tokens).pooler_output.reshape(1,128,768)

    question_emb=torch.cat((question_emb1,question_emb2,question_emb3,question_emb4),dim=0)
    knowledge_tokens=context_tokenizer(list(knowledge),padding=True,return_tensors='pt').to(device)["input_ids"]
    knowledge_emb=context_model(knowledge_tokens).pooler_output.reshape(1,128,768).expand(4,-1,-1)
    
    inbatch_nega=torch.matmul(question_emb,knowledge_emb.permute(0,2,1))
    M=nn.Softmax(dim=0)
    count=0
    for x,i in enumerate(inbatch_nega[0]):
        if x == np.random.choice(128, p=M(i).to('cpu').detach().numpy().copy()):
            count+=1
    M=nn.Softmax(dim=2)
    loss=M(inbatch_nega)
    loss=torch.diagonal(loss, offset=0, dim1=2)+1e-09
    loss = -torch.log(loss)
    loss=torch.mean(loss.view(-1))


    return loss,count
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

def train_dpr(num_epochs=2):
    # loop through all epochs
    optimizer = torch.optim.AdamW(question_model.parameters(), lr=0.0000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = torch.optim.AdamW(context_model.parameters(), lr=0.0000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for i in range(num_epochs):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        print("EPOCH: ", i)
        total_loss = 0.0
        question_model.train()
        context_model.train()
        prev_loss = None
        avg_loss = None
        
        for idx ,(query, knowledge, answer1, answer2, answer3, answer4) in enumerate(tqdm(train_loader, total=len(train_loader), desc="running through training data")):
            optimizer.zero_grad()
            loss,acc = traversal_for_question_context(query, knowledge, answer1, answer2, answer3, answer4)
            train_loss += loss
            train_acc+=acc
            loss.backward()
            optimizer.step()
            avg_train_loss = train_loss / len(train_loader.dataset)  # lossの平均を計算
            avg_train_acc = train_acc / len(train_loader.dataset)
        question_model.eval()
        context_model.eval()
        with torch.no_grad():
            for (query, knowledge, answer1, answer2, answer3, answer4) in valid_loader:
                loss,acc = traversal_for_question_context(query, knowledge, answer1, answer2, answer3, answer4)
                val_loss += loss
                val_acc+=acc
        avg_val_loss = val_loss / len(valid_loader.dataset)
        avg_val_acc = val_acc / len(valid_loader.dataset)

        # print log
        print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                       .format(i+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

        # append list for polt graph after training
        train_loss_list.append(avg_train_loss.detach().cpu().numpy())
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss.detach().cpu().numpy())
        val_acc_list.append(avg_val_acc)
        # print(i)
        if i % 10==0:
            question_model.module.save_pretrained("data_answer_test/model/question_{}".format(i))
            context_model.module.save_pretrained("data_answer_test/model/context_{}".format(i))
        

    # ======== fainal test ======
    question_model.eval()
    context_model.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for (query, knowledge, answer1, answer2, answer3, answer4) in test_loader:  
            loss,acc = traversal_for_question_context(query, knowledge, answer1, answer2, answer3, answer4)
            test_acc+=acc

        print('test_accuracy: {} %'.format(100 * test_acc / len(test_loader.dataset)))
    # save weights
    question_model.module.save_pretrained("data_answer_test/model/question")
    context_model.module.save_pretrained("data_answer_test/model/context")

    
    
train_dpr()