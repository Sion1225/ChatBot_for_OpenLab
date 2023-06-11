import tensorflow as tf
from transformers import MT5Tokenizer, TFMT5ForConditionalGeneration
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Load mT5 model
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

# Encode datas
def convert_datas_to_features(inputs, max_seq_len, tokenizer):
    
    input_ids, attention_masks, = [], []
    
    for i, input in enumerate(tqdm(inputs, total=len(inputs))):

        if input is None or input != input:  # Check if input is None or NaN
            print(f"An error occurred at iteration {i} with input {input}")
            continue

        input_id = tokenizer.encode(input, max_length=max_seq_len, padding="max_length")

        # attention mask (padding mask)
        padding_count = input_id.count(tokenizer.pad_token_id) #pad_token_id: 1
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count #other tokens:1, [pad]:0

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)

    return (input_ids, attention_masks)

# Load JaQuAD DataSet
Data = load_dataset("SkelterLabsInc/JaQuAD")

# Split dataset for train and validation(test)
train = Data['train'].to_pandas()
validation = Data['validation'].to_pandas()

# Split dataset for X and y
X_train = np.array(train["question"])
y_train = train["answers"]
X_test = np.array(validation["question"])
y_test = validation["answers"]

y_train = np.array([item["text"][0] for item in y_train])
y_test = np.array([item["text"][0] for item in y_test])

X_train = ["質問: "+item for item in X_train]
y_train = ["答え: "+item for item in y_train]
X_test = ["質問: "+item for item in X_test]
y_test = ["答え: "+item for item in y_test]
# Set Maximum of sequences length
max_seq_len = 128 #Maximum is 512

# Tokenization (encoding)
X_id_train, X_mask_train = convert_datas_to_features(X_train, max_seq_len=max_seq_len, tokenizer=tokenizer)
y_id_train = np.array([tokenizer.encode(input, max_length=max_seq_len, padding="max_length") for input in y_train], dtype=int)
X_id_test, X_mask_test = convert_datas_to_features(X_test, max_seq_len=max_seq_len, tokenizer=tokenizer)
y_id_test = np.array([tokenizer.encode(input, max_length=max_seq_len, padding="max_length") for input in y_test], dtype=int)

print(X_id_train.shape)
print(X_mask_train.shape)
print(y_id_train.shape)

# Fine-tuning
class CustomT5Model(tf.keras.Model):
    def __init__(self, model_name, *args, **kwargs):
        super(CustomT5Model, self).__init__(*args, **kwargs)
        self.t5 = TFMT5ForConditionalGeneration.from_pretrained(model_name)
    
    def call(self, inputs, training=False):
        return self.t5(*inputs, training=training)
    
    def train_step(self, data):
        data = data[0]
        if len(data) == 3:
            inputs, attention_mask, targets = data
        else:
            raise ValueError("Input data should be a list of [inputs, attention_mask, targets].")

        with tf.GradientTape() as tape:
            outputs = self.t5(inputs, attention_mask=attention_mask, decoder_input_ids=targets, labels=targets)
            loss = outputs.loss
            trainable_vars = self.t5.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {"loss": loss}
'''
    def get_config(self):
        return {"model_name": self.model_name}  # Returns the actual model_name used.

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

# Instantiate the custom model and compile
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = CustomT5Model("google/mt5-base")
model.compile(optimizer=optimizer, loss=loss)
# model fit
model.fit([X_id_train, X_mask_train, y_id_train], epochs=4, batch_size=16, validation_split=0.2)

# model save
model.save("ChatBot_for_OpenLab/Models/mT5_ja", save_format="tf")
'''

# model load
#model = tf.keras.models.load_model("ChatBot_for_OpenLab/Models/mT5_ja", custom_objects={"CustomT5Model": CustomT5Model})
model = TFMT5ForConditionalGeneration.from_pretrained("ChatBot_for_OpenLab/Models/mT5_ja")

# model evaluate
start_tokens = np.ones((len(X_id_test), 1)) * tokenizer.pad_token_id
X_id_test_with_start = np.concatenate([start_tokens, X_id_test[:, :-1]], axis=-1)

results = model.evaluate((X_id_test_with_start, X_mask_test, X_id_test_with_start), np.squeeze(y_id_test))

print(f"Test loss: {results[0]}")

# test
test = [["質問: 日本語話せる？"]]
test = convert_datas_to_features(test, max_seq_len=max_seq_len, tokenizer=tokenizer)
outputs = model.t5.generate(*test)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))