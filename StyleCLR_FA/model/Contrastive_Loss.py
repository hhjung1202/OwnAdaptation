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

    def style_reconstruction(self, content, style, style_label):
        f_c = self.gram_matrix(content) # b*n, ch, ch
        f_s = self.gram_matrix(style) # b, ch, ch
        adaptive_s = f_s[style_label] # b*n, ch, ch
        style_loss = self.MSELoss(f_c, adaptive_s)
        return style_loss * 1e+10

    def style_contrastive_ver1(self, content, style, style_label, b, n, weight=1e+10):
        f_c = F.normalize(self.gram_matrix(content), p=1, dim=-1).view(b,n,-1)             # b, n, ch * ch
        f_s = F.normalize(self.gram_matrix(style), p=1, dim=-1)[style_label].view(b,n,-1)  # b, n, ch * ch

        f_c = f_c.repeat(1, n, 1)                    # b, n*n, -1 AAA_ -> AAA_ AAA_ AAA_
        f_s = f_s.repeat(1, 1, n).view(b, n*n, -1)   # b, n*n, -1 BCD -> BBB CCC DDD

        mse = ((f_c - f_s)**2).sum(dim=2).view(b*n,n) * 1e-1

        label = self.LongTensor([_ for _ in range(n)]).repeat(b)

        # case 1
        style_loss = self.softmin_ce(mse, label) 
        return style_loss

    def style_contrastive_ver2(self, content, style, style_label, b, n, weight=1e+10):
        f_c = F.normalize(self.gram_matrix(content), p=1, dim=-1).view(b,n,-1)             # b, n, ch * ch
        f_s = F.normalize(self.gram_matrix(style), p=1, dim=-1)[style_label].view(b,n,-1)  # b, n, ch * ch

        f_c = f_c.repeat(1, n, 1)                    # b, n*n, -1 AAA_ -> AAA_ AAA_ AAA_
        f_s = f_s.repeat(1, 1, n).view(b, n*n, -1)   # b, n*n, -1 BCD -> BBB CCC DDD

        mse = ((f_c - f_s)**2).sum(dim=2).view(b*n,n) * 1e-1
        
        label = self.LongTensor([_ for _ in range(n)]).repeat(b)

        # case 2
        style_loss = self.softmax_ce_rev(mse, label)
        return style_loss

    def softmin_ce(self, input, target): # y * log(p), p = softmax(-out)
        likelihood = self.softmin(input)
        # print(self.softmin(input * 1e-5).var(dim=1).mean())
        # print(self.softmin(input * 1e-4).var(dim=1).mean())
        # print(self.softmin(input * 1e-3).var(dim=1).mean())
        # print(self.softmin(input * 1e-2).var(dim=1).mean())
        # print(self.softmin(input * 1e-1).var(dim=1).mean())
        # print(self.softmin(input).var(dim=1).mean())
        log_likelihood = likelihood.log()
        nll_loss = F.nll_loss(log_likelihood, target)
        return nll_loss

    def softmax_ce_rev(self, input, target): # y * log(1-p), p = softmax(out)
        likelihood = F.softmax(input, dim=-1)
        # print(F.softmax(input * 1e-5, dim=-1).var(dim=1).mean())
        # print(F.softmax(input * 1e-4, dim=-1).var(dim=1).mean())
        # print(F.softmax(input * 1e-3, dim=-1).var(dim=1).mean())
        # print(F.softmax(input * 1e-2, dim=-1).var(dim=1).mean())
        # print(F.softmax(input * 1e-1, dim=-1).var(dim=1).mean())
        # print(F.softmax(input, dim=-1).var(dim=1).mean())
        log_likelihood_reverse = torch.log(1 - likelihood)
        nll_loss = F.nll_loss(log_likelihood_reverse, target)
        return nll_loss

    def forward(self, content, style, style_label, b, n, L_type="c1"):

        if L_type == "c1":
            style_loss = self.style_contrastive_ver1(content, style, style_label, b, n)
        elif L_type == "c2":
            style_loss = self.style_contrastive_ver2(content, style, style_label, b, n)
        elif L_type == "c3":
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


class Semi_Loss(nn.Module):

    def __init__(self, temperature):
        super(Semi_Loss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, logits, b, n, size_s, y):
        
        content = logits[:-b]
        style = logits[-b:-b+size_s]
        logits_s = content[:n*size_s]
        logits_u = content[n*size_s:].view(-1, n, content.size(-1)) # size_u, n, Cls
        y_ = y.view(-1,1).repeat(1,n).view(-1)
        loss_s = (self.criterion(logits_s, y_) + self.criterion(style, y)) / (n * size_s + b)

        p_u = F.softmax(logits_u, dim=-1)
        y_hat = p_u.mean(dim=1).unsqueeze(1).repeat(1,n,1) # size_u, 1, Cls
        JS_loss = F.kl_div(logits_u, y_hat, reduction="mean")

        pseudo_u = F.softmax(logits_u / self.temperature, dim=-1).mean(dim=1)
        y_hat_ = pseudo_u.unsqueeze(1).repeat(1,n,1).view(-1, content.size(-1)) # size_u * n, Cls
        loss_u = self.soft_label_cross_entropy(logits_u.view(-1, content.size(-1)), y_hat_)

        return loss_s, JS_loss, loss_u

    def soft_label_cross_entropy(self, input, target, eps=1e-5):
        # input (N, C)
        # target (N, C) with soft label
        log_likelihood = input.log_softmax(dim=1)
        soft_log_likelihood = target * log_likelihood
        nll_loss = -torch.sum(soft_log_likelihood.mean(dim=0))
        return nll_loss