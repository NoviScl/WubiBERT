import collections
import json
import os
from copy import deepcopy

from tqdm import tqdm

from ..tools import official_tokenization as tokenization
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
DEBUG = True


def _convert_index(index, pos, M=None, is_start=True):
    if pos >= len(index):
        pos = len(index) - 1
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def read_cmrc_examples(input_file, is_training):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
    train_data = train_data['data']

    # to examples
    examples = []
    mis_match = 0
    for article in tqdm(train_data):
        for para in article['paragraphs']:
            context = para['context']
            
            # Replace special whitespace characters
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

            context_chs = _tokenize_chinese_chars(context)
            doc_tokens = []
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

            for qas in para['qas']:
                qid = qas['id']
                ques_text = qas['question']
                ans_text = qas['answers'][0]['text']
                
                count_i = 0
                start_position = qas['answers'][0]['answer_start']

                end_position = start_position + len(ans_text) - 1
                repeat_limit = 3
                while context[start_position:end_position + 1] != ans_text and count_i < repeat_limit:
                    start_position -= 1
                    end_position -= 1
                    count_i += 1

                while context[start_position] == " " or context[start_position] == "\t" or \
                        context[start_position] == "\r" or context[start_position] == "\n":
                    start_position += 1

                start_position_final = char_to_word_offset[start_position]
                end_position_final = char_to_word_offset[end_position]

                if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                    start_position_final += 1

                actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
                cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

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


def get_subchar_pos(tokens, subchars):
    '''
    Return starting index of each subchar in tokens.
    NOTE: This assumes that the concatenation of tokens is equal to the 
    concatenation of subchars.

    Example:
    >>> Input:
    >>> subchars  = ['jin+', 'ti', 'an+', 'ti', 'an+', 'qi+', 'hen+', 'hao+']
    >>> tokens    = ['jin', '+', 'tian+', 'tian+qi', '+', 'hen+hao+']
    >>> token_pos = [0, 2, 2, 3, 3, 3, 5, 5]
    '''
    if ''.join(tokens) != ''.join(subchars):
        print(tokens)
        print(subchars)
        print('\n\n')
    assert ''.join(tokens) == ''.join(subchars)
    pos = [None] * len(subchars)
    len_t = 0
    len_s = 0
    j = -1  # idx of last token that was added to len_t
    for i, subchar in enumerate(subchars):
        while len_t <= len_s:
            j += 1
            len_t += len(tokens[j])
        pos[i] = j
        len_s += len(subchar)
    return pos


def convert_examples_to_features(*args, **kwargs):
    return _convert_examples_to_features(*args, **kwargs)

