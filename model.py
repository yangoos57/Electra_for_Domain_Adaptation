### Electra-pytorch 라이브러리를 KoElectra에 활용할 수 있도록 일부 변형했습니다.

### Electra로 Domain Adaptation을 수행하기 위해 개발했습니다.

### Generator 모델은 ElectraForMaskedLM로, Discriminator 모델은 ElectraForPreTraining로 불러와야 합니다.

### 더 많은 내용을 알고 싶으신 경우 Domain Adaptation Tutorial을 참고해주세요.

### Electra-pytorh 원본 github 주소 : https://github.com/lucidrains/electra-pytorch


import math
from functools import reduce
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

# constants

Results = namedtuple(
    "Results",
    [
        "loss",
        "mlm_loss",
        "disc_loss",
        "gen_acc",
        "disc_acc",
        "disc_labels",
        "disc_predictions",
    ],
)


# 모델 내부에서 활용되는 함수 정의


def log(t, eps=1e-9):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# main electra class


class Electra(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        tokenizer,
        *,
        num_tokens=35000,
        mask_prob=0.15,
        replace_prob=0.85,
        mask_token_id=4,
        pad_token_id=0,
        mask_ignore_token_ids=[2, 3],
        disc_weight=50.0,
        gen_weight=1.0,
        temperature=1.0,
    ):
        super().__init__()

        """
        num_tokens: 모델 vocab_size
        mask_prob: 토큰 중 [MASK] 토큰으로 대체되는 비율
        replace_prop:  토큰 중 [MASK] 토큰으로 대체되는 비율(?????)
        mask_token_i: [MASK] Token id
        pad_token_i: [PAD] Token id
        mask_ignore_token_id: [CLS],[SEP] Token id
        disc_weigh: discriminator loss의 Weight 조정을 위한 값
        gen_weigh: generator loss의 Weight 조정을 위한 값
        temperature: gumbel_distribution에 활용되는 arg, 값이 높을수록 모집단 분포와 유사한 sampling 수행
        """

        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight

    def forward(self, input_ids, **kwargs):

        input = input_ids["input_ids"]

        # ------ 1단계 Input Data Masking --------#

        """
        - Generator는 Bert와 구조도 동일하고 학습하는 방법도 동일함. 

        - Generator 학습을 위해선 [Masked] 토큰이 필요하므로 input data를 Masking하는 과정이 필요함.

        """

        replace_prob = prob_mask_like(input, self.replace_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # set inverse of mask to padding tokens for labels
        gen_labels = input.masked_fill(~mask, self.pad_token_id)

        # clone the mask, for potential modification if random tokens are involved
        # not to be mistakened for the mask above, which is for all tokens, whether not replaced nor replaced with random tokens
        masking_mask = mask.clone()

        # [mask] input
        masked_input = masked_input.masked_fill(
            masking_mask * replace_prob, self.mask_token_id
        )

        # ------ 2단계 Masking 된 문장을 Generator가 학습하고 가짜 Token을 생성 --------#

        """
        - Generator를 학습하여 MLM_loss 계산(combined_loss 계산에 활용)
        - Generator에서 예측한 문장을 Discriminator 학습에 활용
        - ex) 원본 문장 : ~~~
              마스킹 문장 : 
              가짜 문장 :
        """

        # get generator output and get mlm loss(수정)
        logits = self.generator(masked_input, **kwargs).logits

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2), gen_labels, ignore_index=self.pad_token_id
        )

        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = gumbel_sample(sample_logits, temperature=self.temperature)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float().detach()

        # ------ 3단계 가짜 Token의 진위여부를 Discriminator가 판단하는 단계 --------#

        """
        - 가짜 문장을 학습해 개별 토큰에 대해 진위여부를 판단
        - 진짜 token이라 판단하면 0, 가짜 토큰이라 판단하면 1을 부여
        - 정답과 비교해 disc_loss를 계산(combined_loss 계산에 활용)
        - combined_loss : 학습의 최종 loss임. 모델은 combined_loss의 최솟값을 얻기 위한 방식으로 학습 진행
        """

        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(disc_input, **kwargs).logits
        disc_logits_reshape = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits_reshape[non_padded_indices], disc_labels[non_padded_indices]
        )

        # combined loss 계산
        # disc_weight을 50으로 주는 이유는 discriminator의 task가 복잡하지 않기 떄문임.
        # mlm loss의 경우 vocab_size(=35000) 만큼의 loos 계산을 수행하지만
        # disc_loss의 경우 src_token_len 만큼의 loss 계산을 수행한만큼
        # loss 값에 큰 차이가 발생함. disc_weight은 이를 보완하는 weight임.
        combined_loss = (self.gen_weight * mlm_loss + self.disc_weight * disc_loss,)

        # ------ 모델 성능 및 학습 과정을 추적하기 위한 지표(Metrics) 설계 --------#

        with torch.no_grad():
            # gen mask 예측
            gen_predictions = torch.argmax(logits, dim=-1)

            # fake token 진위 예측
            disc_predictions = torch.round(
                (torch.sign(disc_logits_reshape) + 1.0) * 0.5
            )
            # generator_accuracy
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            # discriminator_accuracy
            disc_acc = (
                0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean()
                + 0.5 * (disc_labels[~mask] == disc_predictions[~mask]).float().mean()
            )

        #

        return Results(
            combined_loss,
            mlm_loss,
            disc_loss,
            gen_acc,
            disc_acc,
            disc_labels,
            disc_predictions,
        )
