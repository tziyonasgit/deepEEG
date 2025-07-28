from Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier

import torch

import unified.utils as utils


def get_models(args):

    num_classes = 0
    use_mean_pooling = True

    use_channels_names = [
        'FP1', 'FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2']

    ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF',
                'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    model = EEGPTClassifier(
        num_classes=num_classes,
        in_channels=len(ch_names),
        img_size=[len(use_channels_names), 2000],
        use_channels_names=use_channels_names,
        use_chan_conv=True,
        use_mean_pooling=use_mean_pooling,)

    return model


if __name__ == '__main__':
    print('hello')

    model = get_models(None)

    print(model)

    finetune = "../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
    model_prefix = ""

    print("Load ckpt from %s" % finetune)

    checkpoint = torch.load(finetune, map_location='cpu', weights_only=False)

    checkpoint_model = checkpoint['state_dict']
    utils.load_state_dict(model, checkpoint_model,
                          prefix=model_prefix)

    print(model)
