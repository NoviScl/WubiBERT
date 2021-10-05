import collections
import copy
import json
import os

from tqdm import tqdm

from ..tools.langconv import Converter
from .utils import (
    _improve_answer_span,
    _check_is_max_context,
    _convert_examples_to_features,
    _is_chinese_char,
    is_fuhao,
    _tokenize_chinese_chars,
    is_whitespace,
)

SPIECE_UNDERLINE = '▁'


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def read_drcd_examples(input_file, is_training, convert_to_simplified, two_level_embeddings):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
    train_data = train_data['data']

    # to examples
    examples = []
    mis_match = 0
    for article in tqdm(train_data):
        for para in article['paragraphs']:
            context = copy.deepcopy(para['context'])
            if two_level_embeddings:
                # Remove weird whitespace
                context = context.replace('\u200b', '')
                context = context.replace(u'\xa0', u'')
                # Adjust answer position accordingly
                for i, qas in enumerate(para['qas']):
                    ans_text = qas['answers'][0]['text']
                    ans_start = qas['answers'][0]['answer_start']
                    if ans_text != context[ans_start:ans_start + len(ans_text)]:
                        lo = None
                        for offset in range(-3, 4):
                            lo = ans_start + offset
                            if context[lo:lo+len(ans_text)] == ans_text:
                                break
                        para['qas'][i]['answers'][0]['answer_start'] = lo
            # 转简体
            if convert_to_simplified:
                context = Traditional2Simplified(context)
            # context中的中文前后加入空格
            context_chs = _tokenize_chinese_chars(context)
            doc_tokens = []
            # ori_doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context_chs:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                if c != SPIECE_UNDERLINE:
                    char_to_word_offset.append(len(doc_tokens) - 1)

            # Generate one example for each question
            for qas in para['qas']:
                qid = qas['id']
                ques_text = qas['question']
                ans_text = qas['answers'][0]['text']
                if convert_to_simplified:
                    ques_text = Traditional2Simplified(ques_text)
                    ans_text = Traditional2Simplified(ans_text)
                start_position_final = None
                end_position_final = None

                # Get start and end position
                start_position = qas['answers'][0]['answer_start']
                end_position = start_position + len(ans_text) - 1

                while context[start_position] == " " or context[start_position] == "\t" or \
                        context[start_position] == "\r" or context[start_position] == "\n":
                    start_position += 1

                start_position_final = char_to_word_offset[start_position]
                end_position_final = char_to_word_offset[end_position]

                if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                    start_position_final += 1
                actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
                cleaned_answer_text = "".join(whitespace_tokenize(ans_text))

                if actual_text != cleaned_answer_text:
                    print(actual_text, 'V.S', cleaned_answer_text)
                    mis_match += 1

                examples.append({'doc_tokens': doc_tokens,
                                 'orig_answer_text': ans_text,
                                 'qid': qid,
                                 'question': ques_text,
                                 'answer': ans_text,
                                 'start_position': start_position_final,
                                 'end_position': end_position_final})

    return examples, mis_match


def convert_examples_to_features(*args, **kwargs):
    return _convert_examples_to_features(*args, **kwargs)
