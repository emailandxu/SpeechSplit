from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
name='wavenet_vocoder',
builder='wavenet',
input_type='raw',
quantize_channels=65536,
sample_rate=16000,
silence_threshold=2,
num_mels=80,
fmin=125,
fmax=7600,
fft_size=1024,
hop_size=256,
frame_shift_ms=None,
min_level_db=-100,
ref_level_db=20,
rescaling=True,
rescaling_max=0.999,
allow_clipping_in_normalization=True,
log_scale_min=-32.23619130191664,
out_channels=30,
layers=24,
stacks=4,
residual_channels=512,
gate_channels=512,
skip_out_channels=256,
dropout=0.050000000000000044,
kernel_size=3,
weight_normalization=True,
legacy=True,
cin_channels=80,
upsample_conditional_features=True,
upsample_scales=[4, 4, 4, 4],
freq_axis_kernel_size=3,
gin_channels=-1,
n_speakers=-1,
pin_memory=True,
test_size=0.0441,
test_num_samples=None,
random_state=1234,
adam_beta1=0.9,
adam_beta2=0.999,
adam_eps=1e-08,
amsgrad=False,
initial_learning_rate=0.001,
lr_schedule='noam_learning_rate_decay',
lr_schedule_kwargs={},
nepochs=2000,
weight_decay=0.0,
clip_thresh=-1,
max_time_sec=None,
max_time_steps=8000,
exponential_moving_average=True,
ema_decay=0.9999,
checkpoint_interval=10000,
train_eval_interval=10000,
test_eval_epoch_interval=5,
save_optimizer_state=True,

    # model   
    freq = 8,
    dim_neck = 8,
    freq_2 = 8,
    dim_neck_2 = 1,
    freq_3 = 8,
    dim_neck_3 = 32,
    
    dim_enc = 512,
    dim_enc_2 = 128,
    dim_enc_3 = 256,
    
    dim_freq = 80,
    dim_spk_emb = 82,
    dim_f0 = 257,
    dim_dec = 512,
    len_raw = 128,
    chs_grp = 16,
    
    # interp
    min_len_seg = 19,
    max_len_seg = 32,
    min_len_seq = 64,
    max_len_seq = 128,
    max_len_pad = 192,
    
    # data loader
    root_dir = 'assets/spmel',
    feat_dir = 'assets/raptf0',
    batch_size = 16,
    mode = 'train',
    shuffle = True,
    num_workers = 0,
    samplier = 8,
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
