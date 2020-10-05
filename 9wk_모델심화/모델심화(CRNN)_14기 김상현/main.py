from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch

from models import CRNN
# from custom_models import CRNN
from utils import CRNN_dataset
from tqdm import tqdm
import argparse
import os


def hyperparameters() :
    """
    argparse는 하이퍼파라미터 설정, 모델 배포 등을 위해 매우 편리한 기능을 제공합니다.
    파이썬 파일을 실행함과 동시에 사용자의 입력에 따라 변수값을 설정할 수 있게 도와줍니다.

    argparse를 공부하여, 아래에 나온 argument를 받을 수 있게 채워주세요.
    해당 변수들은 모델 구현에 사용됩니다.

    ---변수명---
    변수명에 맞춰 type, help, default value 등을 커스텀해주세요 :)

    또한, argparse는 숨겨진 기능이 지이이이인짜 많은데, 다양하게 사용해주시면 우수과제로 가게 됩니다 ㅎㅎ
    """
    wd = os.getcwd()
    parser = argparse.ArgumentParser()
    # ---path--- # 데이터셋의 위치
    parser.add_argument('--path',required=False, default = wd+'/dataset', help = "데이터셋의 위치")
    # ---savepath--- # best model 저장을 위한 파일명
    parser.add_argument('--savepath',default = wd+"/"+'best_model.pth', help = "best model 저장을 위한 파일명")
    # ---batch_size--- # 배치 사이즈
    parser.add_argument('--batch_size', type = int, help="배치 사이즈")
    # ---epochs--- # 에폭 수
    parser.add_argument('--epochs',type = int, help = "에폭 수")
    # ---optim--- # optimizer 선택
    parser.add_argument('--optim', help = "optimizer 선택 (adam/rmsprop)")
    # ---lr--- # learning rate
    parser.add_argument('--lr',type = float, help="학습률")
    # ---device--- # gpu number
    parser.add_argument('--device', help= "gpu 번호")
    # ---img_width--- # 입력 이미지 너비
    parser.add_argument('--img_width',type = int, help="입력 이미지 너비")
    # ---img_height--- # 입력 이미지 높이
    parser.add_argument('--img_height',type = int, help="입력 이미지 높이")

    return parser.parse_args()


def main():
    args = hyperparameters()

    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')

    # gpu or cpu 설정
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # train dataset load
    train_dataset = CRNN_dataset(path=train_path, w=args.img_width, h=args.img_height)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    # test dataset load
    test_dataset = CRNN_dataset(path=test_path, w=args.img_width, h=args.img_height)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


    # model 정의
    model = CRNN(nc=1, nclass =37, nh=256, imgH=args.img_height) #nc =1 ,nclass = 36, nh = 100, #args.img_height

    # loss 정의
    criterion = nn.CTCLoss()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.999))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        assert False, "옵티마이저를 다시 입력해주세요. :("

    model = model.to(device)
    best_test_loss = 100000000
    for i in range(args.epochs):

        print('epochs: ', i)

        print("<----training---->")
        model.train()
        for inputs, targets in tqdm(train_dataloader):
            # ---?--- # inputs의 dimension을 (batch, channel, h, w)로 바꿔주세요. hint: pytorch tensor에 제공되는 함수 사용
            inputs = inputs.permute(0,1,3,2)
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs)
            preds = preds.log_softmax(2)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            """
            CTCLoss의 설명과 해당 로스의 input에 대해 설명해주세요.

            학습데이터에 클래스 라벨만 순서대로 있고 각 클래스의 위치는 어디있는지 모르는 unsegmented
            시퀀스 데이터의 학습을 위해서 사용하는 알고리즘
            ocr(광학 문자 인식)이나 음성 인식등에 널리 사용된다
            input: 예측값, 정답값, 예측 시퀀스의 길이, 정답 시퀀스의 길이

            """

            loss = criterion(preds, target_text, preds_length, target_length) / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print("<----evaluation---->")

        """
        model.train(), model.eval()의 차이에 대해 설명해주세요.
        .eval()을 하는 이유가 무엇일까요?

        모델을 학습할 때 train/eval에 맞게 모델을 변경시킨다
        Dropout이나 batchNormalization을 쓰는 모델은 학습시킬 때와 평가할 때
        구조/역할이 다르기 때문이다.

        """

        model.eval()
        loss = 0.0

        for inputs, targets in tqdm(test_dataloader):
            inputs = inputs.permute(0,1,3,2)
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs)
            preds = preds.log_softmax(2)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            loss += criterion(preds, target_text, preds_length, target_length) / batch_size


        print("test loss: ", loss / len(test_dataloader))
        if loss < best_test_loss:
            # loss가 bset_test_loss보다 작다면 지금의 loss가 best loss가 되겠죠?
            best_test_loss = loss.clone()
            # args.savepath을 이용하여 best model 저장하기
            torch.save(model.state_dict(), args.savepath)
            print("best model 저장 성공")
    #


if __name__=="__main__":
    main()
