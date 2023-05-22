import random
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import random
from model import Model

SAVE_MODEL_PATH = "checkpoint/best_accuracy.pth"

def train(opt):
    device = torch.device("cuda:0" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size)
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # training epoch loop
    best_eval_acc = 0
    start = time.time()
    for ep in range(opt.num_epoch):
        avg_loss = 0
        model.train()
        print(f"{ep + 1}/{opt.num_epoch} epoch start")

        # training mini batch
        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() #ortalama kayıp hesaplama

            if i > 0 and i % 100 == 0:
                print(f"loss:{avg_loss / 100:.4f}") # 100 adımda bir kaybı yazdırma
                avg_loss = 0

        # doğrulama
        if ep % opt.valid_interval == 0: #doğrulama ne zaman yapılacak
            tp, cnt = 0, 0
            model.eval()
            for i, (imgs, labels) in enumerate(testloader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad(): #eğitim verileri korunuyor
                    preds = model(imgs)
                preds = torch.argmax(preds, dim=1)
                tp += (preds == labels).sum().item()
                cnt += labels.shape[0]
            acc = tp / cnt
            print(f"eval acc:{acc:.4f}")

            # modelin kaydedilmesi
            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save(model.state_dict(), SAVE_MODEL_PATH)

        print(f"{ep + 1}/{opt.num_epoch} epoch finished. elapsed time:{time.time() - start:.1f} sec")

    print(f"training finished. best eval acc:{best_eval_acc:.4f}")


random_seed = random.randint(1, 1000)


if __name__ == "__main__":

    # Argümanların belirlenmesi
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation interval")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu if available")
    opt = parser.parse_args()
    print("args", opt)



    opt.manual_seed = random_seed
    # set seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    # training
    train(opt)


####################################python your_script.py --num_epoch 50
##########

    # Örnek tahminler

    device = torch.device("cuda:0" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    # print("device:", device)

    model = Model().to(device)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1)


    # Test veri setini karıştırma
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    indices = list(range(len(testset)))
    random.shuffle(indices)

# Karıştırılmış veri setini kullanarak DataLoader oluşturma
    batch_size = 1
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices))



    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs.data, dim=1)

            pred = predicted[0].item()
            label = labels[0].item()
            print("Predicted:", pred)
            print("Actual:", label)
            break

        image = images[0].cpu().numpy()
        image = image * 0.5 + 0.5
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.show()
