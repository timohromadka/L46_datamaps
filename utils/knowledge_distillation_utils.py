import torch
import torch.nn.functional as F

def kd_training_step(batch, batch_idx, student_model, teacher_model, temperature):
    x, y = batch
    student_logits = student_model(x)
    with torch.no_grad():
        teacher_logits = teacher_model(x)

    KD_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, y)
    loss = KD_loss + hard_loss

    # Logging and other metrics calculation
    acc = student_model.accuracy(torch.argmax(student_logits, dim=1), y)
    student_model.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    student_model.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss


def lsp_training_step(batch, batch_idx, model, label_smoothing):
    x, y = batch
    logits = model(x)

    loss = label_smoothed_nll_loss(logits, y, label_smoothing)
    acc = model.accuracy(torch.argmax(logits, dim=1), y)
    model.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    model.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss


def label_smoothed_nll_loss(lprobs, target, smoothing=0.1):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.mean(dim=-1)
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
    return loss.sum()
