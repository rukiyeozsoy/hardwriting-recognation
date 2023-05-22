import torch
import torch.nn as nn

# Bu kod, PyTorch kullanarak bir derin öğrenme modeli tanımlıyor. Model, 
# Convolutional Neural Network (CNN) olarak adlandırılan bir tür derin öğrenme modelidir ve 
# MNIST veri kümesindeki sayıların tanınması için eğitilmiştir. 
# Model, iki katmanlı bir Convolutional Neural Network (CNN) ve ardından iki tamamen bağlı (fully-connected) katmandan oluşur. 
# Model, ReLU aktivasyon fonksiyonu, MaxPooling, ve Batch Normalization gibi yaygın CNN tekniklerini kullanır. 
# Bu model, görüntü verileri için optimize edilmiştir ve bir adet MNIST veri kümesi üzerinde eğitilmiştir.

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # ReLU aktivasyon fonksiyonu,
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        #Max pooling katmanı
        
        # 32 adet 5x5 filtre kullanılır 
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)

        # 64 adet 5x5 filtre kullanılır 
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)

        #Fully Connected katmanı

        # 7x7 boyutunda 64 adet özellik haritası, 1024 boyutunda bir vektöre dönüştürülür
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1) # 10 rakam için bir olasılık değeri

    # PyTorch ile tanımlanmış bir modelin ileri hesaplama (forward pass) fonksiyonudur. 
    # Bu fonksiyon, modelin girdi verisini alarak çıktıyı hesaplar.

    def forward(self, x):

        # Girdi olarak 28x28 boyutunda bir görüntü tensörü alır. 
        # İlk önce, tensör conv1 katmanından geçirilir, ardından normalleştirme ve ReLU aktivasyon fonksiyonları uygulanır. 
        # Daha sonra, maxpooling işlemi uygulanarak boyut 28x28'den 14x14'e düşürülür.

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 28x28->14x14

        # Aynı işlem conv2 için de tekrarlanır ve sonuçta boyut 7x7 olur. 
        # Son olarak, bu tensör düzleştirilir ve fc1 katmanından geçirilir. 
        # ReLU aktivasyonu sonrasında, tensör fc2 katmanından geçirilir. 
        # Son katman, softmax fonksiyonu ile normalleştirilerek 10 sınıf olasılık değerleri döndürür.

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 14x14->7x7

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        # Yani, bu fonksiyon, girdi olarak bir görüntü tensörü alır ve 10 farklı sınıfa ait olma olasılıklarını hesaplar. 
        # Bu hesaplama sonucunda elde edilen tensör, modelin çıktısıdır.
        
        x = self.fc2(x)
        x = self.softmax(x)

        return x
