import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional

# --- 1. CONFIGURAZIONE PARAMETRI ---
DATA_PATH = "MP_DATA_EMBEDDINGS"           # Cartella contenente le sottocartelle delle glosse
BATCH_SIZE = 128                # Numero di triplette per ogni passo di addestramento
SEQUENCE_LENGTH = 100           # Numero di frame fissi (interpolati)
FEATURE_SIZE = 1662             # Numero di coordinate MediaPipe (Pose+Face+Hands)
EMBEDDING_SIZE = 256            # Dimensione del vettore finale (Embedding)
EPOCHS = 80                     # Numero di cicli di addestramento


if tf.config.list_physical_devices('GPU'):
    print("GPU Metal attivata")
else:
    print("GPU non trovata")

# --- 2. GENERATORE DI DATI (TRIPLET SAMPLING) ---
class SignTripletGenerator(tf.keras.utils.Sequence):
    """
    Genera triplette (Anchor, Positive, Negative) per la Triplet Loss.
    Anchor e Positive appartengono allo stesso segno (es. 'MELA').
    Negative appartiene a un segno diverso (es. 'CASA').
    """
    def __init__(self, base_path, batch_size=BATCH_SIZE):
        self.base_path = base_path
        self.batch_size = batch_size
        self.glosse = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        # Dizionario: Glossa -> Lista di percorsi file .npy
        self.gloss_to_files = {g: [os.path.join(base_path, g, f) for f in os.listdir(os.path.join(base_path, g))] 
                               for g in self.glosse}
        # Teniamo solo le glosse con almeno 2 esempi
        self.glosse = [g for g in self.glosse if len(self.gloss_to_files[g]) >= 2]

    def __len__(self):
        return 100 # Batch per epoca

    def __getitem__(self, index):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            # Campionamento Anchor e Positive
            g_pos = random.choice(self.glosse)
            a_file, p_file = random.sample(self.gloss_to_files[g_pos], 2)
            
            # Campionamento Negative
            g_neg = random.choice([g for g in self.glosse if g != g_pos])
            n_file = random.choice(self.gloss_to_files[g_neg])
            
            anchors.append(np.load(a_file))
            positives.append(np.load(p_file))
            negatives.append(np.load(n_file))
            
        return [np.array(anchors), np.array(positives), np.array(negatives)], np.zeros((self.batch_size,))

# --- 3. ARCHITETTURA DELL'ENCODER LSTM ---
def build_lstm_encoder():
    """
    Crea la rete neurale che trasforma la sequenza video in un embedding.
    Usa LSTM Bi-direzionali per catturare il contesto temporale in entrambi i sensi.
    """
    inputs = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE), name="Input_Video")
    
    # Primo strato: LSTM Bi-direzionale (estrazione feature temporali)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Secondo strato: LSTM (riduzione della sequenza a un singolo vettore stato)
    x = LSTM(256, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Strati densi per la proiezione nello spazio degli embedding
    x = Dense(512, activation='relu')(x)
    outputs = Dense(EMBEDDING_SIZE, activation=None, name="Embedding_Output")(x)
    
    return Model(inputs, outputs, name="LSTM_Encoder")



# --- 4. DEFINIZIONE DELLA TRIPLET LOSS ---
def triplet_loss(y_true, y_pred, margin=0.5):
    """
    Funzione di perdita che minimizza la distanza tra esempi simili 
    e massimizza quella tra esempi diversi.
    """
    batch_size = tf.shape(y_pred)[0] // 3
    anchor = y_pred[0:batch_size]
    positive = y_pred[batch_size:batch_size*2]
    negative = y_pred[batch_size*2:batch_size*3]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

# --- 5. COSTRUZIONE E TRAINING ---
encoder = build_lstm_encoder()

# Modello Siamese: tre input che condividono lo STESSO encoder
input_a = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))
input_p = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))
input_n = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))

emb_a = encoder(input_a)
emb_p = encoder(input_p)
emb_n = encoder(input_n)

out = tf.concat([emb_a, emb_p, emb_n], axis=0)
siamese_model = Model(inputs=[input_a, input_p, input_n], outputs=out)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
siamese_model.compile(optimizer=optimizer, loss=triplet_loss)

# Generatore e avvio
train_gen = SignTripletGenerator(DATA_PATH, batch_size=BATCH_SIZE)

print("\n🚀 Avvio addestramento LSTM...")
siamese_model.fit(
    train_gen, 
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("best_lstm_encoder.h5", save_best_only=True, monitor='loss'),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
    ]
)

# Salvataggio finale dell'encoder per l'inferenza
encoder.save("final_lstm_encoder.h5")
print("\n✅ Modello salvato con successo!")