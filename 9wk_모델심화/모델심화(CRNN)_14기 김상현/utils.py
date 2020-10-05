from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import collections
from glob import glob
import os
from PIL import Image

"""
main.py 함수를 참고하여 다음을 생각해봅시다.

1. CRNN_dataset은 어떤 모듈을 상속받아야 할까요?
- torch.utils.data.Dataset
2. CRNN_dataset의 역할은 무엇일까요? 왜 필요할까요?
- 이미지 데이터를 불러와서 사이즈를 조정하고 label데이터를 생성한다
- 학습을 시키기 위해 고정된 학습 데이터가 필요하기 위해 이를 전처리 해주기 위해 필요하다
3. 1.의 모듈을 상속받는다면 __init__, __len__, __getitem__을 필수로 구현해야 합니다. 각 함수의 역할을 설명해주세요.
    1) 초기화 함수
    2) 길이 반환 함수
    3) 인스턴스의 지정된 인텍스에 해당하는 값을 반환 함수

"""


class CRNN_dataset(Dataset):
    def __init__(self, path, w=100, h=32, alphabet='0123456789abcdefghijklmnopqrstuvwxyz', max_len=36):
        self.max_len=max_len
        self.path = path
        self.files = glob(path+'/*.jpg')
        self.n_image = len(self.files)
        assert (self.n_image > 0), "해당 경로에 파일이 없습니다. :)"

        self.transform = transforms.Compose([
            # ---?---, # image 사이즈를 w, h를 활용하여 바꿔주세요.
            transforms.Resize((w,h)),
            # ---?--- # tensor로 변환해주세요.
            transforms.ToTensor()
        ])
        """
        strLabelConverter의 역할을 설명해주세요.
        1. text 문제를 풀기 위해 해당 함수는 어떤 역할을 하고 있을까요?

        0~9, a~z까지 숫자와 알파벳을 key로 하고 key마다 특정 숫자를 value값으로 하는
        dict를 만든 후 dict를 이용해 encoding/decoding을 수행
        학습울 위해 text를 숫자형 tensor로 변환 시키는 역할

        2. encode, decode의 역할 설명

        encode: 문자열(+iterable object)를 초기화함수에서 정의한 dictionary를 이용해서
        encoding을 수행해서 입력으로 들어온 text를 tensor데이터로 변환하여 변환된 tensor와
        해당 길이를 반환(return)한다
        decode: tensor 타입으로 encode된 text를 본래의 문자열로 반환한다. 연속으로 같은 단어가
        나올 경우 raw=True 지정 필요

        """
        self.converter = strLabelConverter(alphabet)

    def __len__(self):
        return self.n_image # hint: __init__에 정의한 변수 중 하나

    def __getitem__(self,idx):
        label = self.files[idx].split('_')[1]
        img = Image.open(self.files[idx]).convert('L')
        img = self.transform(img)
        """
        max_len이 왜 필요할까요? # hint: text data라는 점

        같은 길이의 시퀀스가 입력으로 필요하다
        max_len보다 작을 겨우 1로 남은 자리 패딩

        """

        if len(label) > self.max_len:
            label = label[:self.max_len]
        label_text, label_length = self.converter.encode(label)

        if len(label_text) < self.max_len:
            temp = torch.ones(self.max_len-len(label), dtype=torch.int)
            label_text = torch.cat([label_text, temp])

        # return ---?---, (---?---, ---?---) # hint: main.py를 보면 알 수 있어요 :)
        return img, (label_text, label_length)



# 아래 함수는 건드리지 마시고, 그냥 쓰세요 :)
class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
