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

    def style_reconstruction(self, content, style, style_label):
        f_c = self.gram_matrix(content) # b*n, ch, ch
        f_s = self.gram_matrix(style) # b, ch, ch
        adaptive_s = f_s[style_label] # b*n, ch, ch
        style_loss = self.MSELoss(f_c, adaptive_s)
        return style_loss

    def style_contrastive_ver1(self, content, style, style_label, b, n):
        f_c = self.gram_matrix(content).view(n,b,-1)             # n, b, ch * ch
        f_s = self.gram_matrix(style)[style_label].view(n,b,-1)  # n, b, ch * ch

        f_c = f_c.transpose(0,1).repeat(1, n, 1)                    # b, n*n, -1 AAA_ -> AAA_ AAA_ AAA_
        f_s = f_s.transpose(0,1).repeat(1, 1, n).view(b, n*n, -1)   # b, n*n, -1 BCD -> BBB CCC DDD

        mse = ((f_c - f_s)**2).sum(dim=2).view(b*n,n)

        # case 1
        style_loss = self.softmin_ce(mse, style_label)

        return style_loss

    def style_contrastive_ver2(self, content, style, style_label, b, n):
        f_c = self.gram_matrix(content).view(n,b,-1)             # n, b, ch * ch
        f_s = self.gram_matrix(style)[style_label].view(n,b,-1)  # n, b, ch * ch

        f_c = f_c.transpose(0,1).repeat(1, n, 1)                    # b, n*n, -1 AAA_ -> AAA_ AAA_ AAA_
        f_s = f_s.transpose(0,1).repeat(1, 1, n).view(b, n*n, -1)   # b, n*n, -1 BCD -> BBB CCC DDD

        mse = ((f_c - f_s)**2).sum(dim=2).view(b*n,n)

        # case 2
        style_loss = self.softmax_ce_rev(mse, style_label)

        return style_loss

    def softmin_ce(self, input, target): # y * log(p), p = softmax(-out)
        log_likelihood = self.softmin(input).log()
        nll_loss = F.nll_loss(log_likelihood, target)
        return nll_loss

    def softmax_ce_rev(self, input, target): # y * log(1-p), p = softmax(out)
        log_likelihood_reverse = torch.log(1 - self.softmax(input))
        nll_loss = F.nll_loss(log_likelihood_reverse, target)
        return nll_loss

    def forward(self, content, style, style_label, b, n, L_type="c1"):

        if L_type is "c1":
            style_loss = self.style_contrastive_ver1(content, style, style_label, b, n)
        elif L_type is "c2":
            style_loss = self.style_contrastive_ver2(content, style, style_label, b, n)
        else:
            style_loss = self.style_reconstruction(content, style, style_label)

        return style_loss
    
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