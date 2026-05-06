import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional

# --- 1. CONFIGURAZIONE PARAMETRI ---
DATA_PATH = "MP_DATA_EMBEDDINGS"
BATCH_SIZE = 128
BATCH = 256;
SEQUENCE_LENGTH = 125
FEATURE_SIZE = 1662
EMBEDDING_SIZE = 256
EPOCHS = 60

# --- GPU CHECK ---
if tf.config.list_physical_devices('GPU'):
    print("GPU attivata")
else:
    print("GPU non trovata")


# --- 2. GENERATORE TRIPLET ---
class SignTripletGenerator(tf.keras.utils.Sequence):

    def __init__(self, base_path, batch_size=BATCH_SIZE):
        self.base_path = base_path
        self.batch_size = batch_size

        self.glosse = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]

        self.gloss_to_files = {
            g: [os.path.join(base_path, g, f)
                for f in os.listdir(os.path.join(base_path, g))]
            for g in self.glosse
        }

        self.glosse = [g for g in self.glosse if len(self.gloss_to_files[g]) >= 2]

    def __len__(self):
        return BATCH

    def __getitem__(self, index):
        anchors, positives, negatives = [], [], []

        for _ in range(self.batch_size):

            # anchor + positive
            g_pos = random.choice(self.glosse)
            a_file, p_file = random.sample(self.gloss_to_files[g_pos], 2)

            # negative 
            g_neg = random.choice(self.glosse)
            while g_neg == g_pos:
                g_neg = random.choice(self.glosse)

            n_file = random.choice(self.gloss_to_files[g_neg])

            anchors.append(np.load(a_file))
            positives.append(np.load(p_file))
            negatives.append(np.load(n_file))
            
        # FIX: Restituiamo una tupla per gli input (x) e un array di zeri fittizio per (y)
        # La loss triplet non utilizza y_true, quindi possiamo restituire un array di zeri o qualsiasi cosa di dimensione (batch_size,)
        x = (np.array(anchors), np.array(positives), np.array(negatives))
        y = np.zeros((self.batch_size))
        return x, y


# --- 3. ENCODER LSTM ---
def build_lstm_encoder():

    inputs = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))

    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(256, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation='relu')(x)
    x = Dense(EMBEDDING_SIZE, activation=None)(x)

    # normalizzazione
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1)
    )(x)

    return Model(inputs, outputs, name="LSTM_Encoder")


# --- 4. TRIPLET LOSS CORRETTA ---
def triplet_loss(margin=0.2):
    def loss(y_true, y_pred):

        # FIX: Slicing del tensore concatenato sull'asse 1 per recuperare i 3 embedding
        anchor = y_pred[:, :EMBEDDING_SIZE]
        positive = y_pred[:, EMBEDDING_SIZE:EMBEDDING_SIZE*2]
        negative = y_pred[:, EMBEDDING_SIZE*2:]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        return tf.reduce_mean(
            tf.maximum(pos_dist - neg_dist + margin, 0.0)
        )

    return loss


# --- 5. MODELLO SIAMESE ---
encoder = build_lstm_encoder()

input_a = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))
input_p = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))
input_n = Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE))

emb_a = encoder(input_a)
emb_p = encoder(input_p)
emb_n = encoder(input_n)

# FIX: Concatenazione lungo l'asse delle feature (axis=1) invece dell'asse del batch (axis=0)
merged = tf.keras.layers.Concatenate(axis=1)([emb_a, emb_p, emb_n])
siamese_model = Model(inputs=[input_a, input_p, input_n], outputs=merged)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

siamese_model.compile(
    optimizer=optimizer,
    loss=triplet_loss()
)


# --- 6. TRAINING ---
train_gen = SignTripletGenerator(DATA_PATH, batch_size=BATCH_SIZE)

print("\n🚀 Avvio training...")

siamese_model.fit(
    train_gen,
    epochs=EPOCHS,

    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            "best_lstm_encoder.h5",
            save_best_only=True,
            monitor='loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=6,
            restore_best_weights=True
        )
    ]
)


# --- 7. SALVATAGGIO ---
encoder.save("final_lstm_encoder.h5")

print("\n✅ Modello salvato!")