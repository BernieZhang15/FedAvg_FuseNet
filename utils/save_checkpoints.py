import os
import torch


def save_checkpoints(states, args, is_best):
    print('[PROGRESS] Saving the model', end="", flush=True)
    checkpoint_dir = os.path.join(args.checkpoints_dir, args.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if is_best:
        best_model_filename = os.path.join(checkpoint_dir, 'best_model.pth.tar')
        torch.save(states, best_model_filename)
        print('\r[INFO] Best model has been successfully updated: %s' % best_model_filename)
        return

    checkpoint_filename = os.path.join(checkpoint_dir, 'model_checkpoint_{}'.format(states['epoch']) + '.pth.tar')
    torch.save(states, checkpoint_filename)
    print('\r[INFO] Checkpoint has been saved: %s' % checkpoint_filename)
