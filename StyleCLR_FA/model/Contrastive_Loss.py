import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers

class Content_Contrastive(nn.Module):

    def __init__(self, temperature):
        super(Content_Contrastive, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    
    def _get_label(self, b, n):
        # label gen : (b*n)
        label = self.LongTensor([[_] for _ in range(b)]).repeat(1,n).view(-1)
        return label

    def _cosine_similarity(self, content, style):
        # content size : (b*n, 1, dim)
        # style size : (1, b, dim)
        # v size : (b*n, b)
        v = self.similarity(content.unsqueeze(1), style.unsqueeze(0))
        return v

    def forward(self, content, style, b, n):
        
        logits = self._cosine_similarity(content, style) # size : (b*n, b)
        logits /= self.temperature # softmax temperature
        labels = self._get_label(b, n)
        loss = self.criterion(logits, labels)

        return loss / (n * b)


class Style_Contrastive(nn.Module):
    def __init__(self):
        super(Style_Contrastive, self).__init__()
        self.MSELoss = nn.MSELoss()
        self.softmin = nn.Softmin(dim=-1)
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    def style_contrastive(self, content, style, b, n):
        f_c = F.normalize(self.gram_matrix(content), p=1, dim=-1).view(b,1,-1)             # b, n, ch * ch
        f_s = F.normalize(self.gram_matrix(style), p=1, dim=-1).view(1,b,-1)  # b, n, ch * ch

        mse = ((f_c - f_s)**2).sum(dim=2).view(b,b)
        return mse

    def forward(self, content, style, b, n):
        mse, label = self.style_contrastive(content, style, b, n)
        return mse, label
    
    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a, b, c * d)
        G = torch.bmm(features, torch.transpose(features, 1,2))
        return G.div(b * c * d)

    # def gram_matrix2(input):
    #     a, b, c, d = input.size()
    #     features = input.view(a, b, c * d)
    #     G = torch.bmm(torch.transpose(features, 1,2), features)
    #     return G.div(b * c * d)
