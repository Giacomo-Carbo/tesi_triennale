import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional


# --- 1. CONFIGURAZIONE PARAMETRI ---
DATA_PATH = "MP_DATA_EMBEDDINGS"
BATCH_SIZE = 2048  # Aumentato a 1024 per sfruttare meglio la GPU, ma dipende dalla memoria disponibile
STEPS_PER_EPOCH = 200  # Aumentato per sfruttare bene il batch grande
VAL_STEPS = 40         # Step per la validazione
SEQUENCE_LENGTH = 125
FEATURE_SIZE = 258
EMBEDDING_SIZE = 256
EPOCHS = 100            # Aumentato, l'Early Stopping fermerà il training al momento giusto

# --- GPU CHECK ---
if tf.config.list_physical_devices('GPU'):
    print("GPU attivata")
else:
    print("GPU non trovata")


# --- 2. GENERATORE TRIPLET OTTIMIZZATO (CON CACHE E SPLIT) ---
class SignTripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_path, batch_size=BATCH_SIZE, steps=100, is_validation=False, split_ratio=0.85):
        self.base_path = base_path
        self.batch_size = batch_size
        self.steps = steps
        self.cache = {} # Mantiene i file in RAM dopo la prima lettura
        
        all_glosse = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        self.gloss_to_files = {}
        for g in all_glosse:
            files = [os.path.join(base_path, g, f) for f in os.listdir(os.path.join(base_path, g))]
            if len(files) < 2:
                continue
                
            # Split Train/Val deterministico
            split_idx = int(len(files) * split_ratio)
            if is_validation:
                self.gloss_to_files[g] = files[split_idx:]
            else:
                self.gloss_to_files[g] = files[:split_idx]

        # Rimuoviamo classi che dopo lo split non hanno abbastanza campioni
        self.glosse = [g for g in self.gloss_to_files.keys() if len(self.gloss_to_files[g]) >= 2]

    def _load_data(self, filepath):
        # Carica il file dal disco solo se non è già presente in RAM
        if filepath not in self.cache:
            self.cache[filepath] = np.load(filepath)
        return self.cache[filepath]

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        anchors, positives, negatives = [], [], []

        for _ in range(self.batch_size):
            # Anchor + Positive
            g_pos = random.choice(self.glosse)
            a_file, p_file = random.sample(self.gloss_to_files[g_pos], 2)

            # Negative 
            g_neg = random.choice(self.glosse)
            while g_neg == g_pos:
                g_neg = random.choice(self.glosse)
            n_file = random.choice(self.gloss_to_files[g_neg])

            # Usiamo il metodo _load_data per sfruttare la cache
            anchors.append(self._load_data(a_file))
            positives.append(self._load_data(p_file))
            negatives.append(self._load_data(n_file))
            
        x = (np.array(anchors), np.array(positives), np.array(negatives))
        y = np.zeros((self.batch_size,))
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

    # Normalizzazione L2 (necessaria per la cosine similarity)
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1)
    )(x)

    return Model(inputs, outputs, name="LSTM_Encoder")


# --- 4. COSINE SIMILARITY LOSS ---
def triplet_cosine_loss(margin=0.2):
    def loss(y_true, y_pred):
        anchor = y_pred[:, :EMBEDDING_SIZE]
        positive = y_pred[:, EMBEDDING_SIZE:EMBEDDING_SIZE*2]
        negative = y_pred[:, EMBEDDING_SIZE*2:]

        pos_sim = tf.reduce_sum(anchor * positive, axis=1)
        neg_sim = tf.reduce_sum(anchor * negative, axis=1)

        return tf.reduce_mean(
            tf.maximum(neg_sim - pos_sim + margin, 0.0)
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

merged = tf.keras.layers.Concatenate(axis=1)([emb_a, emb_p, emb_n])
siamese_model = Model(inputs=[input_a, input_p, input_n], outputs=merged)

# Learning rate scalato per il batch da 1024
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0008)  # Aumentato da 0.0005 a 0.0008 per batch più grande

siamese_model.compile(
    optimizer=optimizer,
    loss=triplet_cosine_loss()
)


# --- 6. TRAINING ---
# Creazione dei due generatori distinti
train_gen = SignTripletGenerator(DATA_PATH, batch_size=BATCH_SIZE, steps=STEPS_PER_EPOCH, is_validation=False)
val_gen = SignTripletGenerator(DATA_PATH, batch_size=BATCH_SIZE, steps=VAL_STEPS, is_validation=True)

print("\nAvvio training...")

siamese_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            "best_lstm_encoder.keras", # Aggiornato al formato keras e con nome LSTM
            save_best_only=True,
            monitor='val_loss',        # Monitoriamo la validazione
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True
        )
    ]
)


# --- 7. SALVATAGGIO ---
encoder.save("final_lstm_encoder1.keras")

print("\nModello salvato con successo!")