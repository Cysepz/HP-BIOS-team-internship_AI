import torch
import torch.nn as nn

import torch.optim as optim
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

# Define the transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layer, num_decoder_layer):
        super(TransformerModel, self).__init__()
        # Embedding layer: 將輸入的整數序列轉換為 d_model 維度的向量表示
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Transformer layer: model 的核心，接受 input 的 embedding vector；執行 self-attention 和 positional-encoding
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layer, num_decoder_layer)
        # Linear layer: 將 model 的輸出映射到 vocab_size 維度的向量，藉此生成模型的輸出預測
        self.fc = nn.Linear(d_model, vocab_size)
    
    # Forward function 定義了資料在模型中的流動方向（src: input sequence, tgt: target sequence）
    def forward(self, src, tgt):
        # 先將 src & tgt 進行 embedding
        src_embedding = self.embedding(src)
        tgt_embedding = self.embedding(tgt)
        # 將 embedding vector 丟入 transformer layer 進行運算
        output = self.transformer(src_embedding, tgt_embedding)
        # 交由 linear layer 生成模型的輸出（輸出是對 tgt 的預測）
        output = self.fc(output)
        return output
    

# 使用 torchtext 中的 datasets 來下載並讀取 IMDB 資料集
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors=GloVe(name='6B', dim=100))
LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)

INPUT_DIM = len(TEXT.vocab)
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

model = TransformerModel(INPUT_DIM, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS).to(device) # 建立Transformer模型並移至GPU（如果可用）

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad() # 每次迭代前先將 optimizer 的 gradient 歸零
        src = batch.text
        tgt = batch.label.unsqueeze(1) # 將 label 轉為二維張量
        output = model(src, tgt) # 將 src & tgt 丟入模型進行運算（模型向前傳播）
        loss = criterion(output, tgt) # 計算 loss
        loss.backward()
        optimizer.step() # 更新權重
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src = batch.text
            tgt = batch.label.unsqueeze(1)
            output = model(src, tgt)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}')