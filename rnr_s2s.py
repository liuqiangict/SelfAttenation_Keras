
import random
import numpy as np

from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import tensorflow as tf

from DataLoader import TokenList
from DataLoader import pad_to_longest

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

class Encoder():
    def __init__(self, itoken_num, latent_dim, layers=3):
        self.emb_layer = Embedding(itoken_num, latent_dim, mask_zero=True)
        cells = [GRUCell(latent_dim) for _ in range(layers)]
        self.rnn_layer = RNN(cells, return_state=True)

    def __call__(self, x):
        x = self.emb_layer(x)
        xh = self.rnn_layer(x)
        x, h = xh[0], xh[1:]
        return x, h

class Decoder():
    def __init__(self, otoken_num, latent_dim, layers=3):
        self.emb_layer = Embedding(otoken_num, latent_dim, mask_zero=True)
        cells = [GRUCell(latent_dim) for _ in range(layers)]
        self.rnn_layer = RNN(cells, return_sequences=True, return_state=True)
        self.out_layer = Dense(otoken_num)

    def __call__(self, x, state):
        x = self.emb_layer(x)
        xh = self.rnn_layer(x, initial_state=state)
        x, h = xh[0], xh[1:]
        x = TimeDistributed(self.out_layer)(x)
        return x, h


class RNNSeq2Seq:
    def __init__(self, itokens, otokens, latent_dim, layers = 3):
        self.itokens = itokens
        self.otokens = otokens

        encoder_inputs = Input(shape = (None,), dtype = 'int32')
        decoder_inputs = Input(shape = (None,), dtype = 'int32')

        encoder = Encoder(itokens.num(), latent_dim, layers)
        decoder = Decoder(otokens.num(), latent_dim, layers)

        encoder_output, encoder_states = encoder(encoder_inputs)

        dinputs = Lambda(lambda x : x[:, :-1])(decoder_inputs)
        dtargets = Lambda(lambda x : x[:, 1:])(decoder_inputs)
        decoder_outputs, decoder_state_h = decoder(dinputs, encoder_states)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)
        
        loss = Lambda(get_loss)([decoder_outputs, dtargets])
        self.ppl = Lambda(K.exp)(loss)
        self.accu = Lambda(get_accu)([decoder_outputs, dtargets])
        
        self.model = Model([encoder_inputs, decoder_inputs], loss)
        self.model.add_loss([K.mean(loss)])
        
        encoder_model = Model(encoder_inputs, encoder_states)
        
        decoder_states_inputs = [Input(shape=(latent_dim,)) for _ in range(3)]
        
        decoder_outputs, decoder_states = decoder(decoder_inputs, decoder_states_inputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def compile(self, optimizer):
        self.model.compile(optimizer, None)
        self.model.metrics_names.append('ppl')
        self.model.metrics_tensors.append(self.ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)

    def decode_sequence(self, input_seq, delimiter=''):
        input_mat = np.zeros((1, len(input_seq)+3))
        input_mat[0,0] = self.itokens.id('<S>')
        for i, z in enumerate(input_seq): 
            input_mat[0,1+i] = self.itokens.id(z)
        input_mat[0,len(input_seq)+1] = self.itokens.id('</S>')

        state_value = self.encoder_model.predict_on_batch(input_mat)
        target_seq = np.zeros((1, 1))
        target_seq[0,0] = self.otokens.id('<S>')

        decoded_tokens = []
        while True:
            output_tokens_and_h = self.decoder_model.predict_on_batch([target_seq] + state_value)
            output_tokens, h = output_tokens_and_h[0], output_tokens_and_h[1:]
            sampled_token_index = np.argmax(output_tokens[0,-1,:])
            sampled_token = self.otokens.token(sampled_token_index)
            decoded_tokens.append(sampled_token)
            if sampled_token == '</S>' or len(decoded_tokens) > 50:
                break
            target_seq = np.zeros((1, 1))
            target_seq[0,0] = sampled_token_index
            state_value = h

        return delimiter.join(decoded_tokens[:-1])



if __name__ == '__main__':
    itokens = TokenList(list('0123456789'))
    otokens = TokenList(list('0123456789abcdefx'))
    
    def GenSample():
        x = random.randint(0, 99999)
        y = hex(x)
        x = str(x)
        return x, y

    X, Y = [], []
    for _ in range(100000):
        x, y = GenSample()
        X.append(list(x))
        Y.append(list(y))

    X, Y = pad_to_longest(X, itokens), pad_to_longest(Y, otokens)
    print(X.shape, Y.shape)
    
    s2s = RNNSeq2Seq(itokens, otokens, 128)

    plot_model(to_file='model.png', model=s2s.model, show_shapes=True)
    plot_model(to_file='encoder.png',model=s2s.encoder_model, show_shapes=True)
    plot_model(to_file='decoder.png',model=s2s.decoder_model, show_shapes=True)


    s2s.compile('rmsprop')
    s2s.model.summary()
    '''
    class TestCallback(Callback):
        def on_epoch_end(self, epoch, logs = None):
            print('\n')
            for test in [123, 12345, 34567]:
                ret = s2s.decode_sequence(str(test))
                print(test, ret, hex(test))
            print('\n')

    #s2s.model.load_weights('model.h5')
    s2s.model.fit([X, Y], None, batch_size=1024, epochs=10, validation_split=0.05, callbacks=[TestCallback()])
    s2s.model.save_weights('model.h5')
    '''