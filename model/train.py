import torch
import math

from progress_bar import ProgressBar

def train(train_dataloader, train_size, inputs_labels_func,
          model, criterion, optimizer, scheduler=None,
          device=torch.device("cpu"), num_epochs=100, pbar_len=80, do_carriage_return=True,
          eval_func=None, eval_interval=10):
    pbar_train_len = pbar_len
    if not do_carriage_return:
        ProgressBar.print_total_line(pbar_train_len)
        print()

    epoch = 0
    while epoch < num_epochs:
        # Eval
        if eval_func is not None and epoch % eval_interval == 0:  # eval_interval - 1:
            torch.set_grad_enabled(False)
            eval_func(model, inputs_labels_func)
            torch.set_grad_enabled(True)

        # Train
        pbar = ProgressBar(len(train_dataloader), length=pbar_train_len,
                           do_carriage_return=do_carriage_return)
        pbar.start(front_msg="Train ")

        model.train()
        optimizer.zero_grad()
        train_loss_sum = torch.zeros(1, device=device)
        for data in train_dataloader:
            optimizer.zero_grad()
            inputs, labels = inputs_labels_func(data)
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            train_loss_sum += train_loss.detach()
            train_loss.backward()
            optimizer.step()
            pbar.update(front_msg="Train ")
        pbar.reset()

        # Print and record stastistics

        avg_train_loss = (train_loss_sum.cpu() / train_size).item()
        print("[epoch %2d]  train loss: %.5f" % (epoch + 1, avg_train_loss))

        if scheduler is not None:
            scheduler.step()

        epoch += 1