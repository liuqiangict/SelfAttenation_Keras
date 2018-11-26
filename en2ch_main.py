import os
import sys
import DataLoader as reader
from keras.optimizers import *
from keras.callbacks import *

from Transformer import Transformer
from Transformer import LRSchedulerPerStep
from Transformer import LRSchedulerPerEpoch

'''
itokens, otokens = reader.MakeS2SDict('.\data\en2ch.s2s.txt', dict_file='.\data\en2ch_word.txt')
Xtrain, Ytrain = reader.MakeS2SData('.\data\en2ch.s2s.txt', itokens, otokens, h5_file='.\data\en2ch.h5')
Xvalid, Yvalid = reader.MakeS2SData('.\data\en2ch.s2s.valid.txt', itokens, otokens, h5_file='.\data\en2ch.valid.h5')
'''

itokens, otokens = reader.MakeS2SDict('.\data\en2ch.s2s.txt')
Xtrain, Ytrain = reader.MakeS2SData('.\data\en2ch.s2s.txt', itokens, otokens)
Xvalid, Yvalid = reader.MakeS2SData('.\data\en2ch.s2s.valid.txt', itokens, otokens)

print('Seq 1 words : ', itokens.num())
print('Seq 2 words : ', otokens.num())
print('Train shapes : ', Xtrain.shape, Ytrain.shape)
print('Valid shapes : ', Xvalid.shape, Yvalid.shape)


d_model = 512
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)
mfile = '.\model\en2ch.model.h5'

ls_scheduler = LRSchedulerPerStep(d_model, 4000)
lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.1, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()
try: 
    s2s.model.load_weights(mfile)
except Exception as e:
    print(e)
    print('\n\nnew model')


for i in range(30):
    s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=1, validation_data=([Xvalid, Yvalid], None), callbacks=[ls_scheduler, model_saver])
    print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=''))

while True:
    quest = input('> ')
    print(s2s.decode_sequence_fast(quest.split(), delimiter=''))
    rets = s2s.beam_search(quest.split(), delimiter='')
    for x, y in rets:
        print(x, y)
